from __future__ import annotations

import json
import re
import secrets
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .guard import (
    GuardDecision,
    GuardRunner,
    evaluate_guard,
    hours_text,
    list_active_gpu_kernels,
    parse_hours,
    read_gpu_quota,
    snapshot_hash,
)
from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file, utc_now
from .kaggle import redact_diagnostic
from .ledger import EventType, ExperimentEvent, Ledger, reconstruct_runs, start_payload
from .preflight import validate_preflight


_KERNEL_REF = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_AUTHORIZATION_ID = re.compile(r"^auth-[a-f0-9]{32}$")
PushRunner = Callable[[Sequence[str]], Any]


class LaunchError(ValueError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise LaunchError("AUTHORIZATION_MALFORMED", "authorization timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise LaunchError("AUTHORIZATION_MALFORMED", "authorization timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _kernel_directory(workspace_root: Path, path: str | Path, kernel_ref: str) -> Path:
    if not _KERNEL_REF.fullmatch(kernel_ref):
        raise LaunchError("KERNEL_REF_INVALID", "kernel ref must be owner/slug")
    root = workspace_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate == root or root not in candidate.parents or not candidate.is_dir():
        raise LaunchError("KERNEL_PATH_INVALID", "kernel directory must be inside the workspace")
    metadata_path = candidate / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise LaunchError("KERNEL_METADATA_INVALID", "kernel-metadata.json is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError("KERNEL_METADATA_INVALID", "kernel metadata is unreadable") from exc
    if not isinstance(metadata, Mapping) or metadata.get("id") != kernel_ref:
        raise LaunchError("KERNEL_REF_MISMATCH", "kernel metadata id does not match kernel ref")
    if metadata.get("enable_gpu") is not True:
        raise LaunchError("KERNEL_GPU_DISABLED", "guarded GPU launch requires enable_gpu=true")
    return candidate


def hash_kernel_directory(directory: Path) -> str:
    records: list[dict[str, str]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise LaunchError("KERNEL_PATH_INVALID", f"kernel source may not contain symlinks: {path}")
        if path.is_file() and ".ipynb_checkpoints" not in path.parts:
            records.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    if not records:
        raise LaunchError("KERNEL_SOURCE_EMPTY", "kernel directory has no source files")
    return sha256_bytes(canonical_json_bytes(records))


def validate_research_refresh(
    workspace_root: Path,
    config: Mapping[str, Any],
    *,
    now: datetime,
) -> None:
    """Require a recent source-based notebook audit before spending GPU."""

    policy = config.get("research_refresh_policy")
    if not isinstance(policy, Mapping):
        return
    try:
        max_age = int(policy["max_audit_age_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchError(
            "RESEARCH_POLICY_INVALID", "research audit age policy is malformed"
        ) from exc
    if max_age <= 0:
        raise LaunchError(
            "RESEARCH_POLICY_INVALID", "research audit age must be positive"
        )
    audit_path = workspace_root.resolve() / "policies" / "notebook_audits.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audited_at = _parse_time(str(audit["audited_at"]))
        discussions_audited_at = _parse_time(str(audit["discussions_audited_at"]))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, LaunchError) as exc:
        raise LaunchError(
            "RESEARCH_AUDIT_UNAVAILABLE", "source-based notebook audit is unavailable"
        ) from exc
    if policy.get("require_source_review") is True:
        notebooks = audit.get("notebooks")
        source_reviews = [
            row.get("reviewed_sha256")
            for row in notebooks
            if isinstance(row, Mapping)
        ] if isinstance(notebooks, list) else []
        if not any(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in source_reviews
        ):
            raise LaunchError(
                "RESEARCH_AUDIT_UNAVAILABLE",
                "research audit has no hash-bound notebook source review",
            )
    current = now.astimezone(timezone.utc)
    if max(audited_at, discussions_audited_at) > current + timedelta(minutes=5):
        raise LaunchError(
            "RESEARCH_AUDIT_FUTURE", "notebook audit timestamp is in the future"
        )
    if any(
        current - timestamp > timedelta(hours=max_age)
        for timestamp in (audited_at, discussions_audited_at)
    ):
        raise LaunchError(
            "RESEARCH_AUDIT_STALE", "refresh notebooks and discussions before GPU launch"
        )


@dataclass(frozen=True)
class LaunchAuthorization:
    authorization_id: str
    run_id: str
    kernel_ref: str
    kernel_directory: str
    kernel_source_sha256: str
    declared_max_runtime_hours: str
    reserve_hours: str
    quota_snapshot_hash: str
    preflight_report: str
    preflight_report_sha256: str
    issued_at: str
    expires_at: str
    nonce: str
    schema_version: int = 1
    authorization_sha256: str = ""

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        kernel_ref: str,
        kernel_directory: str,
        kernel_source_sha256: str,
        declared_max_runtime_hours: str,
        reserve_hours: str,
        quota_snapshot_hash: str,
        preflight_report: str,
        preflight_report_sha256: str,
        now: datetime,
        ttl_seconds: int,
    ) -> "LaunchAuthorization":
        if ttl_seconds <= 0 or ttl_seconds > 1800:
            raise LaunchError("AUTHORIZATION_TTL_INVALID", "authorization TTL must be 1..1800 seconds")
        authorization = cls(
            authorization_id=f"auth-{uuid.uuid4().hex}",
            run_id=run_id,
            kernel_ref=kernel_ref,
            kernel_directory=kernel_directory,
            kernel_source_sha256=kernel_source_sha256,
            declared_max_runtime_hours=declared_max_runtime_hours,
            reserve_hours=reserve_hours,
            quota_snapshot_hash=quota_snapshot_hash,
            preflight_report=preflight_report,
            preflight_report_sha256=preflight_report_sha256,
            issued_at=_utc_text(now),
            expires_at=_utc_text(now + timedelta(seconds=ttl_seconds)),
            nonce=secrets.token_urlsafe(24),
        )
        return cls(**{**authorization.__dict__, "authorization_sha256": authorization.computed_sha256()})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LaunchAuthorization":
        try:
            authorization = cls(**{key: value[key] for key in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise LaunchError("AUTHORIZATION_MALFORMED", "authorization fields are missing") from exc
        if authorization.schema_version != 1 or not _AUTHORIZATION_ID.fullmatch(
            authorization.authorization_id
        ):
            raise LaunchError("AUTHORIZATION_MALFORMED", "authorization identity is invalid")
        return authorization

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if key != "authorization_sha256"
        }

    def computed_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.unsigned_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "authorization_sha256": self.authorization_sha256}


def authorization_path(workspace_root: Path, authorization_id: str) -> Path:
    if not _AUTHORIZATION_ID.fullmatch(authorization_id):
        raise LaunchError("AUTHORIZATION_ID_INVALID", "authorization ID is invalid")
    return workspace_root.resolve() / ".biohub" / "authorizations" / f"{authorization_id}.json"


def consumed_path(workspace_root: Path, authorization_id: str) -> Path:
    return authorization_path(workspace_root, authorization_id).with_suffix(".consumed.json")


def load_authorization(workspace_root: Path, authorization_id: str) -> LaunchAuthorization:
    path = authorization_path(workspace_root, authorization_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaunchError("AUTHORIZATION_NOT_FOUND", "authorization was not found") from exc
    except json.JSONDecodeError as exc:
        raise LaunchError("AUTHORIZATION_MALFORMED", "authorization JSON is malformed") from exc
    if not isinstance(value, Mapping):
        raise LaunchError("AUTHORIZATION_MALFORMED", "authorization must be an object")
    authorization = LaunchAuthorization.from_dict(value)
    if not secrets.compare_digest(
        authorization.authorization_sha256, authorization.computed_sha256()
    ):
        raise LaunchError("AUTHORIZATION_CHANGED", "authorization content hash changed")
    return authorization


def authorize_launch(
    *,
    workspace_root: Path,
    ledger: Ledger,
    runner: GuardRunner,
    config: Mapping[str, Any],
    run_id: str,
    kernel_directory: str | Path,
    kernel_ref: str,
    preflight_report: str | Path,
    now: datetime | None = None,
    ttl_seconds: int = 600,
) -> tuple[LaunchAuthorization, GuardDecision]:
    root = workspace_root.resolve()
    current = (now or utc_now()).astimezone(timezone.utc)
    validate_research_refresh(root, config, now=current)
    runs = reconstruct_runs(ledger.read_events())
    state = runs.get(run_id)
    if state is None:
        raise LaunchError("RUN_NOT_REGISTERED", "experiment is not registered")
    runtime = state.registered.get("declared_max_runtime_hours")
    directory = _kernel_directory(root, kernel_directory, kernel_ref)
    report = validate_preflight(
        preflight_report,
        root,
        run_id=run_id,
        declared_max_runtime_hours=runtime,
        now=current,
    )
    active = list_active_gpu_kernels(runner, str(config["slug"]))
    quota = read_gpu_quota(runner)
    decision = evaluate_guard(
        run_id=run_id,
        registered_status=state.status,
        registered_declared_runtime=runtime,
        declared_max_runtime=runtime,
        quota=quota,
        active_kernels=active,
        reserve=config["gpu_reserve_hours"],
        notebook_maximum=config["notebook_runtime_limit_hours"],
    )
    ledger.append(ExperimentEvent.create(run_id, EventType.GUARD_DECISION, decision.to_dict()))
    if not decision.authorized:
        code = decision.reason_codes[0]
        raise LaunchError(code, f"launch authorization rejected: {', '.join(decision.reason_codes)}")
    report_path = Path(preflight_report)
    if not report_path.is_absolute():
        report_path = root / report_path
    report_path = report_path.resolve()
    authorization = LaunchAuthorization.create(
        run_id=run_id,
        kernel_ref=kernel_ref,
        kernel_directory=directory.relative_to(root).as_posix(),
        kernel_source_sha256=hash_kernel_directory(directory),
        declared_max_runtime_hours=hours_text(parse_hours(runtime)),
        reserve_hours=hours_text(parse_hours(config["gpu_reserve_hours"])),
        quota_snapshot_hash=decision.quota_snapshot_hash or "",
        preflight_report=report_path.relative_to(root).as_posix(),
        preflight_report_sha256=report.report_sha256,
        now=current,
        ttl_seconds=ttl_seconds,
    )
    atomic_write_json(authorization_path(root, authorization.authorization_id), authorization.to_dict())
    return authorization, decision


def validate_authorization(
    authorization: LaunchAuthorization,
    *,
    workspace_root: Path,
    ledger: Ledger,
    runner: GuardRunner,
    config: Mapping[str, Any],
    nonce: str,
    now: datetime | None = None,
    expected_run_id: str | None = None,
    expected_kernel_directory: str | Path | None = None,
) -> tuple[GuardDecision, Decimal]:
    root = workspace_root.resolve()
    current = (now or utc_now()).astimezone(timezone.utc)
    validate_research_refresh(root, config, now=current)
    if not secrets.compare_digest(authorization.authorization_sha256, authorization.computed_sha256()):
        raise LaunchError("AUTHORIZATION_CHANGED", "authorization content hash changed")
    if current < _parse_time(authorization.issued_at) - timedelta(seconds=60):
        raise LaunchError("AUTHORIZATION_NOT_YET_VALID", "authorization issue time is in the future")
    if current >= _parse_time(authorization.expires_at):
        raise LaunchError("AUTHORIZATION_EXPIRED", "authorization has expired")
    if consumed_path(root, authorization.authorization_id).exists():
        raise LaunchError("AUTHORIZATION_CONSUMED", "authorization is single-use and already consumed")
    if not secrets.compare_digest(str(nonce), authorization.nonce):
        raise LaunchError("AUTHORIZATION_NONCE_MISMATCH", "confirmation nonce does not match")
    if expected_run_id is not None and expected_run_id != authorization.run_id:
        raise LaunchError("AUTHORIZATION_RUN_MISMATCH", "authorization belongs to another run")
    directory = _kernel_directory(root, authorization.kernel_directory, authorization.kernel_ref)
    if expected_kernel_directory is not None:
        expected = Path(expected_kernel_directory)
        if not expected.is_absolute():
            expected = root / expected
        if expected.resolve() != directory:
            raise LaunchError("AUTHORIZATION_PATH_MISMATCH", "authorization binds another path")
    if not secrets.compare_digest(hash_kernel_directory(directory), authorization.kernel_source_sha256):
        raise LaunchError("KERNEL_SOURCE_CHANGED", "kernel source changed after authorization")
    runs = reconstruct_runs(ledger.read_events())
    state = runs.get(authorization.run_id)
    if state is None:
        raise LaunchError("RUN_NOT_REGISTERED", "authorized run no longer exists")
    registered_runtime = state.registered.get("declared_max_runtime_hours")
    if parse_hours(registered_runtime) != parse_hours(authorization.declared_max_runtime_hours):
        raise LaunchError("DECLARED_RUNTIME_MISMATCH", "registered runtime changed")
    if parse_hours(config["gpu_reserve_hours"]) != parse_hours(authorization.reserve_hours):
        raise LaunchError("RESERVE_POLICY_CHANGED", "reserve policy changed after authorization")
    report = validate_preflight(
        root / authorization.preflight_report,
        root,
        run_id=authorization.run_id,
        declared_max_runtime_hours=authorization.declared_max_runtime_hours,
        now=current,
    )
    if not secrets.compare_digest(report.report_sha256, authorization.preflight_report_sha256):
        raise LaunchError("PREFLIGHT_CHANGED", "preflight report changed after authorization")
    active = list_active_gpu_kernels(runner, str(config["slug"]))
    quota = read_gpu_quota(runner)
    decision = evaluate_guard(
        run_id=authorization.run_id,
        registered_status=state.status,
        registered_declared_runtime=registered_runtime,
        declared_max_runtime=authorization.declared_max_runtime_hours,
        quota=quota,
        active_kernels=active,
        reserve=authorization.reserve_hours,
        notebook_maximum=config["notebook_runtime_limit_hours"],
    )
    if not decision.authorized:
        raise LaunchError(decision.reason_codes[0], ", ".join(decision.reason_codes))
    if not secrets.compare_digest(snapshot_hash(quota, active), authorization.quota_snapshot_hash):
        raise LaunchError("KAGGLE_STATE_CHANGED", "quota or active-kernel state changed")
    return decision, quota.remaining


def default_push_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), shell=False, capture_output=True, text=True, check=False
    )


def push_kernel(
    authorization: LaunchAuthorization,
    *,
    workspace_root: Path,
    ledger: Ledger,
    runner: GuardRunner,
    config: Mapping[str, Any],
    nonce: str,
    execute: bool,
    push_runner: PushRunner = default_push_runner,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not execute:
        return {"executed": False, "authorization": authorization.to_dict()}
    decision, quota_before = validate_authorization(
        authorization,
        workspace_root=workspace_root,
        ledger=ledger,
        runner=runner,
        config=config,
        nonce=nonce,
        now=now,
    )
    root = workspace_root.resolve()
    directory = _kernel_directory(root, authorization.kernel_directory, authorization.kernel_ref)
    try:
        atomic_write_json(
            consumed_path(root, authorization.authorization_id),
            {
                "schema_version": 1,
                "authorization_id": authorization.authorization_id,
                "run_id": authorization.run_id,
                "consumed_at": _utc_text((now or utc_now()).astimezone(timezone.utc)),
                "authorization_sha256": authorization.authorization_sha256,
            },
        )
    except FileExistsError as exc:
        raise LaunchError(
            "AUTHORIZATION_CONSUMED", "authorization was consumed by another launcher"
        ) from exc
    command = ["kaggle", "kernels", "push", "-p", str(directory)]
    try:
        result = push_runner(command)
        return_code = getattr(result, "returncode", 0)
        if return_code not in (None, 0):
            diagnostic = redact_diagnostic(
                str(getattr(result, "stderr", "") or getattr(result, "stdout", "")).strip()
            )
            suffix = f": {diagnostic}" if diagnostic else ""
            raise RuntimeError(f"kaggle kernels push exited {return_code}{suffix}")
    except Exception as exc:
        ledger.append(
            ExperimentEvent.create(
                authorization.run_id,
                EventType.LAUNCH_FAILED,
                {
                    "authorization_id": authorization.authorization_id,
                    "reason": f"{type(exc).__name__}: {exc}"[:2000],
                },
            )
        )
        raise LaunchError("KERNEL_PUSH_FAILED", "kernel push failed; authorization was consumed") from exc
    started = ExperimentEvent.create(
        authorization.run_id,
        EventType.STARTED,
        start_payload(
            kaggle_ref=authorization.kernel_ref,
            authorization_id=authorization.authorization_id,
            quota_before_hours=quota_before,
        ),
    )
    ledger.append(started)
    return {
        "executed": True,
        "authorization_id": authorization.authorization_id,
        "run_id": authorization.run_id,
        "command": command,
        "started_event_id": started.event_id,
        "guard": decision.to_dict(),
    }
