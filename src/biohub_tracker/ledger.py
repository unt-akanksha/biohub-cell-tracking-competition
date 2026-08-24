from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file, utc_now


class LedgerError(RuntimeError):
    pass


class LedgerLockTimeout(LedgerError):
    pass


class LedgerCorruptionError(LedgerError):
    pass


class TransitionError(LedgerError):
    pass


PRODUCER_REGISTRATION_FIELDS = (
    "manifest_sha256",
    "fold_id",
    "train_membership_sha256",
    "calibration_membership_sha256",
    "evaluation_membership_sha256",
    "model_sha256",
    "config_sha256",
    "code_sha256",
    "data_sha256",
)
PRODUCER_TERMINAL_FIELDS = (
    "evidence_eligible",
    "graph_inventory_sha256",
    "artifact_hashes",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EventType(StrEnum):
    REGISTERED = "registered"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    DECISION = "decision"
    AMENDMENT = "amendment"
    GUARD_DECISION = "guard_decision"
    LAUNCH_FAILED = "launch_failed"


class RunStatus(StrEnum):
    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExperimentEvent:
    event_id: str
    run_id: str
    event_type: EventType
    created_at: str
    payload: Mapping[str, Any]
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        run_id: str,
        event_type: EventType | str,
        payload: Mapping[str, Any],
        *,
        created_at: str | None = None,
        event_id: str | None = None,
    ) -> "ExperimentEvent":
        return cls(
            event_id=event_id or f"evt-{uuid.uuid4().hex}",
            run_id=_bounded_text(run_id, "run_id", 160),
            event_type=EventType(event_type),
            created_at=created_at or _iso_utc(utc_now()),
            payload=dict(payload),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentEvent":
        required = {"schema_version", "event_id", "run_id", "event_type", "created_at", "payload"}
        if not required <= value.keys() or not isinstance(value.get("payload"), Mapping):
            raise ValueError("event is missing required fields")
        event = cls(
            schema_version=int(value["schema_version"]),
            event_id=_bounded_text(value["event_id"], "event_id", 160),
            run_id=_bounded_text(value["run_id"], "run_id", 160),
            event_type=EventType(value["event_type"]),
            created_at=_bounded_text(value["created_at"], "created_at", 80),
            payload=dict(value["payload"]),
        )
        if event.schema_version != 1:
            raise ValueError(f"unsupported event schema version: {event.schema_version}")
        _parse_time(event.created_at)
        return event

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        return value


@dataclass
class RunState:
    run_id: str
    status: RunStatus
    registered: dict[str, Any]
    parent: str | None
    events: list[ExperimentEvent] = field(default_factory=list)
    terminal: dict[str, Any] | None = None
    decision: str | None = None
    decision_evidence: list[str] = field(default_factory=list)
    amendments: list[dict[str, Any]] = field(default_factory=list)
    guard_decisions: list[dict[str, Any]] = field(default_factory=list)
    launch_failures: list[dict[str, Any]] = field(default_factory=list)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp: {value!r}") from exc


def _bounded_text(value: Any, name: str, limit: int, *, required: bool = True) -> str:
    text = str(value).strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    if len(text) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    return text


def decimal_text(value: Any, name: str, *, positive: bool = False) -> str:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not result.is_finite() or (positive and result <= 0):
        comparator = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {comparator}")
    return format(result, "f")


def generate_run_id(slug: str, *, now: datetime | None = None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", slug.casefold()).strip("-") or "experiment"
    timestamp = (now or utc_now()).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{normalized[:48]}-{secrets.token_hex(4)}"


def workspace_file(workspace_root: Path, path: str | Path) -> Path:
    root = workspace_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise ValueError(f"artifact path must be an existing file inside the workspace: {path}")
    return candidate


def artifact_record(
    workspace_root: Path, path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, str]:
    candidate = workspace_file(workspace_root, path)
    digest = sha256_file(candidate)
    if expected_sha256 and not secrets.compare_digest(digest, expected_sha256.casefold()):
        raise ValueError(f"artifact hash mismatch for {path}")
    return {
        "path": candidate.relative_to(workspace_root.resolve()).as_posix(),
        "sha256": digest,
    }


def git_state(workspace_root: Path) -> dict[str, Any]:
    def run(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=workspace_root,
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )

    head_result = run(["rev-parse", "HEAD"])
    status_result = run(["status", "--porcelain=v1", "--untracked-files=normal"])
    head = head_result.stdout.strip() if head_result.returncode == 0 else "unavailable"
    status = status_result.stdout if status_result.returncode == 0 else "unavailable"
    return {
        "git_head": head,
        "dirty": bool(status.strip()) if status != "unavailable" else None,
        "dirty_state_sha256": sha256_bytes(status.encode("utf-8")),
    }


def _event_order(event: ExperimentEvent) -> tuple[Any, ...]:
    precedence = {
        EventType.REGISTERED: 0,
        EventType.GUARD_DECISION: 1,
        EventType.LAUNCH_FAILED: 1,
        EventType.STARTED: 2,
        EventType.COMPLETED: 3,
        EventType.FAILED: 3,
        EventType.REJECTED: 3,
        EventType.DECISION: 4,
        EventType.AMENDMENT: 5,
    }
    return (_parse_time(event.created_at), precedence[event.event_type], event.event_id)


def reconstruct_runs(events: Iterable[ExperimentEvent]) -> dict[str, RunState]:
    grouped: dict[str, list[ExperimentEvent]] = {}
    for event in events:
        grouped.setdefault(event.run_id, []).append(event)
    runs: dict[str, RunState] = {}
    for run_id in sorted(grouped):
        ordered = sorted(grouped[run_id], key=_event_order)
        registered = [event for event in ordered if event.event_type is EventType.REGISTERED]
        if len(registered) != 1:
            raise TransitionError(f"run {run_id} must have exactly one registered event")
        registration = dict(registered[0].payload)
        state = RunState(
            run_id=run_id,
            status=RunStatus.REGISTERED,
            registered=registration,
            parent=registration.get("parent"),
            events=ordered,
        )
        seen_event_ids: set[str] = {registered[0].event_id}
        for event in ordered:
            if event.event_type is EventType.REGISTERED:
                continue
            if event.event_type is EventType.STARTED:
                if state.status is not RunStatus.REGISTERED:
                    raise TransitionError(f"run {run_id} has an illegal repeated/late start")
                state.status = RunStatus.RUNNING
            elif event.event_type in {EventType.COMPLETED, EventType.FAILED, EventType.REJECTED}:
                if state.status is not RunStatus.RUNNING:
                    raise TransitionError(f"run {run_id} has a terminal event before/after running")
                state.status = RunStatus(event.event_type.value)
                state.terminal = dict(event.payload)
            elif event.event_type is EventType.DECISION:
                if state.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.REJECTED}:
                    raise TransitionError(f"run {run_id} has a decision before a terminal event")
                state.decision = str(event.payload.get("decision"))
                state.decision_evidence = [str(item) for item in event.payload.get("evidence", [])]
            elif event.event_type is EventType.AMENDMENT:
                if event.payload.get("target_event_id") not in seen_event_ids:
                    raise TransitionError(f"run {run_id} amendment targets a later or unknown event")
                state.amendments.append(dict(event.payload))
            elif event.event_type is EventType.GUARD_DECISION:
                state.guard_decisions.append(dict(event.payload))
            elif event.event_type is EventType.LAUNCH_FAILED:
                state.launch_failures.append(dict(event.payload))
            seen_event_ids.add(event.event_id)
        runs[run_id] = state
    for state in runs.values():
        if state.parent and state.parent not in runs:
            raise TransitionError(f"run {state.run_id} has unknown parent {state.parent}")
    return runs


def validate_transition(events: Sequence[ExperimentEvent], event: ExperimentEvent) -> None:
    if any(existing.event_id == event.event_id for existing in events):
        raise TransitionError(f"duplicate event ID: {event.event_id}")
    runs = reconstruct_runs(events) if events else {}
    existing = runs.get(event.run_id)
    if event.event_type is EventType.REGISTERED:
        if existing:
            raise TransitionError(f"duplicate run ID: {event.run_id}")
        parent = event.payload.get("parent")
        if parent and parent not in runs:
            raise TransitionError(f"unknown parent run: {parent}")
        _bounded_text(event.payload.get("hypothesis"), "hypothesis", 1000)
        decimal_text(event.payload.get("declared_max_runtime_hours"), "declared runtime", positive=True)
        validate_producer_registration_evidence(event.payload)
        return
    if not existing:
        raise TransitionError(f"unknown run ID: {event.run_id}")
    if event.event_type is EventType.STARTED and existing.status is not RunStatus.REGISTERED:
        raise TransitionError(f"run {event.run_id} cannot start from {existing.status}")
    if event.event_type is EventType.STARTED:
        _bounded_text(event.payload.get("kaggle_ref"), "kaggle ref", 240)
        _bounded_text(event.payload.get("authorization_id"), "authorization ID", 240)
        decimal_text(event.payload.get("quota_before_hours"), "quota before")
    if event.event_type in {EventType.COMPLETED, EventType.FAILED, EventType.REJECTED}:
        if existing.status is not RunStatus.RUNNING:
            raise TransitionError(f"run {event.run_id} cannot finish from {existing.status}")
        decimal_text(event.payload.get("actual_runtime_hours"), "actual runtime", positive=True)
        decimal_text(event.payload.get("quota_after_hours"), "quota after")
        if event.event_type is EventType.COMPLETED:
            validate_complete_metrics(event.payload.get("metrics"))
            validate_producer_terminal_evidence(event.payload)
        elif event.event_type is EventType.FAILED:
            _bounded_text(event.payload.get("failure_reason"), "failure reason", 2000)
        else:
            _bounded_text(event.payload.get("failed_gate"), "failed gate", 500)
            _bounded_text(event.payload.get("reason"), "rejection reason", 2000)
    if event.event_type is EventType.DECISION:
        if existing.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.REJECTED}:
            raise TransitionError(f"run {event.run_id} cannot be decided from {existing.status}")
        decision = str(event.payload.get("decision"))
        if decision not in {"promote", "retain", "retire", "inconclusive"}:
            raise TransitionError(f"invalid decision: {decision}")
        evidence = event.payload.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise TransitionError("decision requires evidence references")
        if decision == "promote" and all(_is_public_only_evidence(item) for item in evidence):
            raise TransitionError("promotion requires non-public-score evidence")
    if event.event_type is EventType.AMENDMENT:
        target = event.payload.get("target_event_id")
        if not any(candidate.event_id == target for candidate in existing.events):
            raise TransitionError(f"unknown amendment target: {target}")
        _bounded_text(event.payload.get("correction_reason"), "correction reason", 1000)
        if not isinstance(event.payload.get("replacement_fields"), Mapping):
            raise TransitionError("amendment replacement_fields must be an object")
    if event.event_type is EventType.GUARD_DECISION:
        if existing.status is not RunStatus.REGISTERED:
            raise TransitionError(
                f"run {event.run_id} cannot receive launch authorization from {existing.status}"
            )
        if not isinstance(event.payload.get("reason_codes"), list):
            raise TransitionError("guard decision requires reason_codes")
    if event.event_type is EventType.LAUNCH_FAILED:
        if existing.status is not RunStatus.REGISTERED:
            raise TransitionError(f"run {event.run_id} cannot record launch failure from {existing.status}")
        _bounded_text(event.payload.get("authorization_id"), "authorization ID", 240)
        _bounded_text(event.payload.get("reason"), "launch failure reason", 2000)


class _ExclusiveLock:
    def __init__(self, path: Path, timeout_seconds: float):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.descriptor: int | None = None

    def __enter__(self) -> "_ExclusiveLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.descriptor, f"pid={os.getpid()}\n".encode("ascii"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LedgerLockTimeout(f"timed out waiting for ledger lock: {self.path}")
                time.sleep(0.01)

    def __exit__(self, *_: Any) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


@dataclass
class Ledger:
    path: Path
    workspace_root: Path
    lock_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        self.path = self.path.resolve()
        self.workspace_root = self.workspace_root.resolve()
        if self.workspace_root not in self.path.parents:
            raise ValueError("ledger path must be inside the workspace")

    @property
    def lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    def _quarantine(self, fragment: bytes) -> Path:
        directory = self.workspace_root / ".biohub" / "quarantine"
        digest = sha256_bytes(fragment)
        target = directory / f"ledger-tail-{digest[:16]}.bin"
        directory.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(fragment)
                handle.flush()
                os.fsync(handle.fileno())
        return target

    def _decode(self, data: bytes, *, quarantine: bool = True) -> list[ExperimentEvent]:
        if not data:
            return []
        lines = data.splitlines(keepends=True)
        events: list[ExperimentEvent] = []
        for index, line in enumerate(lines):
            final = index == len(lines) - 1
            if not line.endswith(b"\n"):
                target = self._quarantine(line) if quarantine else None
                raise LedgerCorruptionError(f"truncated ledger tail quarantined at {target}")
            try:
                value = json.loads(line)
                event = ExperimentEvent.from_dict(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if final:
                    target = self._quarantine(line) if quarantine else None
                    raise LedgerCorruptionError(f"malformed ledger tail quarantined at {target}") from exc
                raise LedgerCorruptionError(f"malformed ledger event on line {index + 1}") from exc
            events.append(event)
        reconstruct_runs(events)
        return events

    def read_events(self) -> list[ExperimentEvent]:
        if not self.path.exists():
            return []
        return self._decode(self.path.read_bytes())

    def append(self, event: ExperimentEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _ExclusiveLock(self.lock_path, self.lock_timeout_seconds):
            existing_bytes = self.path.read_bytes() if self.path.exists() else b""
            events = self._decode(existing_bytes)
            validate_transition(events, event)
            payload = canonical_json_bytes(event.to_dict()) + b"\n"
            with self.path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    def repair_truncated(self, reason: str) -> Path:
        reason = _bounded_text(reason, "repair reason", 500)
        with _ExclusiveLock(self.lock_path, self.lock_timeout_seconds):
            data = self.path.read_bytes()
            try:
                self._decode(data, quarantine=False)
            except LedgerCorruptionError:
                pass
            else:
                raise LedgerError("ledger has no malformed trailing fragment")
            lines = data.splitlines(keepends=True)
            valid_lines: list[bytes] = []
            bad_fragment = b""
            for index, line in enumerate(lines):
                try:
                    if not line.endswith(b"\n"):
                        raise ValueError("truncated")
                    ExperimentEvent.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    if index != len(lines) - 1:
                        raise LedgerCorruptionError("repair refuses corruption before the final line")
                    bad_fragment = line
                    break
                valid_lines.append(line)
            if not bad_fragment:
                raise LedgerError("ledger has no repairable trailing fragment")
            quarantine_path = self._quarantine(bad_fragment)
            temporary = self.path.with_suffix(self.path.suffix + ".repair.tmp")
            with temporary.open("wb") as handle:
                handle.write(b"".join(valid_lines))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            acknowledgement = {
                "schema_version": 1,
                "repaired_at": _iso_utc(utc_now()),
                "reason": reason,
                "quarantine_path": quarantine_path.relative_to(self.workspace_root).as_posix(),
                "fragment_sha256": sha256_bytes(bad_fragment),
            }
            ack_path = quarantine_path.with_suffix(".ack.json")
            if ack_path.exists():
                ack_path = quarantine_path.with_name(
                    f"{quarantine_path.stem}-{secrets.token_hex(2)}.ack.json"
                )
            return atomic_write_json(ack_path, acknowledgement)


def registration_payload(
    *,
    hypothesis: str,
    parent: str | None,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    split: str,
    declared_max_runtime_hours: Any,
    code: Mapping[str, Any],
    data_artifact: Mapping[str, Any] | None = None,
    model_artifact: Mapping[str, Any] | None = None,
    producer_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hypothesis = _bounded_text(hypothesis, "hypothesis", 1000)
    runtime = decimal_text(declared_max_runtime_hours, "declared runtime", positive=True)
    payload = {
        "hypothesis": hypothesis,
        "parent": parent,
        "config": dict(config),
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "code": dict(code),
        "data_artifact": dict(data_artifact) if data_artifact else None,
        "model_artifact": dict(model_artifact) if model_artifact else None,
        "seeds": [int(seed) for seed in seeds],
        "split": _bounded_text(split, "split", 200),
        "declared_max_runtime_hours": runtime,
        "authorized_for_submission": False,
    }
    if producer_evidence is not None:
        normalized = validate_producer_registration_evidence(producer_evidence, required=True)
        if normalized["config_sha256"] != payload["config_sha256"]:
            raise ValueError("producer evidence config_sha256 does not match config")
        payload.update(normalized)
    return payload


def _sha256_text(value: Any, name: str) -> str:
    digest = str(value).casefold()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _artifact_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("artifact_hashes must be a non-empty object")
    result: dict[str, str] = {}
    for name in sorted(value, key=str):
        key = _bounded_text(name, "artifact hash name", 240)
        result[key] = _sha256_text(value[name], f"artifact_hashes.{key}")
    return result


def validate_producer_registration_evidence(
    payload: Mapping[str, Any], *, required: bool = False
) -> dict[str, Any] | None:
    """Validate the additive model-producer lineage projection.

    A legacy registration contains ``config_sha256`` but none of the new
    identity fields. It remains byte-for-byte readable and is deliberately
    evidence-ineligible.
    """

    extension_fields = set(PRODUCER_REGISTRATION_FIELDS) - {"config_sha256"}
    present = extension_fields.intersection(payload)
    if not present and not required:
        return None
    missing = set(PRODUCER_REGISTRATION_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"producer registration evidence missing {sorted(missing)}")
    result = {
        name: _sha256_text(payload[name], name)
        for name in PRODUCER_REGISTRATION_FIELDS
        if name != "fold_id"
    }
    result["fold_id"] = _bounded_text(payload["fold_id"], "fold_id", 200)
    return {name: result[name] for name in PRODUCER_REGISTRATION_FIELDS}


def validate_producer_terminal_evidence(
    payload: Mapping[str, Any], *, required: bool = False
) -> dict[str, Any] | None:
    present = set(PRODUCER_TERMINAL_FIELDS).intersection(payload)
    if not present and not required:
        return None
    missing = set(PRODUCER_TERMINAL_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"producer terminal evidence missing {sorted(missing)}")
    if payload["evidence_eligible"] is not True:
        raise ValueError("evidence_eligible must be true when terminal evidence is supplied")
    return {
        "evidence_eligible": True,
        "graph_inventory_sha256": _sha256_text(
            payload["graph_inventory_sha256"], "graph_inventory_sha256"
        ),
        "artifact_hashes": _artifact_hashes(payload["artifact_hashes"]),
    }


def normalize_exact_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): normalize_exact_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_exact_values(item) for item in value]
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    return value


def validate_complete_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise ValueError("completed experiment requires a metrics object")
    pooled = metrics.get("pooled")
    required_pooled = {
        "adjusted_edge_jaccard",
        "edge_jaccard",
        "division_jaccard",
        "node_recall",
    }
    if not isinstance(pooled, Mapping) or not required_pooled <= pooled.keys():
        raise ValueError(f"metrics.pooled requires {sorted(required_pooled)}")
    counts = metrics.get("division_counts")
    if not isinstance(counts, Mapping) or not {"tp", "fp", "fn"} <= counts.keys():
        raise ValueError("metrics.division_counts requires tp, fp, and fn")
    for key in ("by_embryo", "by_fold"):
        if not isinstance(metrics.get(key), Mapping):
            raise ValueError(f"metrics.{key} must be an object")
    if "worst_movie_delta" not in metrics:
        raise ValueError("metrics.worst_movie_delta is required")
    return normalize_exact_values(metrics)


def start_payload(
    *, kaggle_ref: str, authorization_id: str, quota_before_hours: Any
) -> dict[str, Any]:
    return {
        "kaggle_ref": _bounded_text(kaggle_ref, "kaggle ref", 240),
        "authorization_id": _bounded_text(authorization_id, "authorization ID", 240),
        "quota_before_hours": decimal_text(quota_before_hours, "quota before"),
    }


def completed_payload(
    *,
    actual_runtime_hours: Any,
    quota_after_hours: Any,
    metrics: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]] = (),
    reports: Sequence[Mapping[str, Any]] = (),
    public_score: Any | None = None,
    producer_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "actual_runtime_hours": decimal_text(
            actual_runtime_hours, "actual runtime", positive=True
        ),
        "quota_after_hours": decimal_text(quota_after_hours, "quota after"),
        "metrics": validate_complete_metrics(metrics),
        "artifacts": [dict(item) for item in artifacts],
        "reports": [dict(item) for item in reports],
        "public_score": decimal_text(public_score, "public score") if public_score is not None else None,
        "authorized_for_submission": False,
    }
    if producer_evidence is not None:
        payload.update(validate_producer_terminal_evidence(producer_evidence, required=True))
    return payload


def failed_payload(
    *,
    actual_runtime_hours: Any,
    quota_after_hours: Any,
    failure_reason: str,
    traceback_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "actual_runtime_hours": decimal_text(
            actual_runtime_hours, "actual runtime", positive=True
        ),
        "quota_after_hours": decimal_text(quota_after_hours, "quota after"),
        "failure_reason": _bounded_text(failure_reason, "failure reason", 2000),
        "traceback_artifact": dict(traceback_artifact) if traceback_artifact else None,
        "authorized_for_submission": False,
    }


def rejected_payload(
    *,
    actual_runtime_hours: Any,
    quota_after_hours: Any,
    failed_gate: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "actual_runtime_hours": decimal_text(
            actual_runtime_hours, "actual runtime", positive=True
        ),
        "quota_after_hours": decimal_text(quota_after_hours, "quota after"),
        "failed_gate": _bounded_text(failed_gate, "failed gate", 500),
        "reason": _bounded_text(reason, "rejection reason", 2000),
        "authorized_for_submission": False,
    }


def _is_public_only_evidence(value: Any) -> bool:
    normalized = str(value).casefold().replace("-", "_")
    return normalized.startswith("public") or "leaderboard" in normalized


def decision_payload(decision: str, evidence: Sequence[str]) -> dict[str, Any]:
    normalized = str(decision).casefold()
    if normalized not in {"promote", "retain", "retire", "inconclusive"}:
        raise ValueError(f"invalid decision: {decision}")
    references = [_bounded_text(item, "evidence reference", 500) for item in evidence]
    if not references:
        raise ValueError("decision requires at least one evidence reference")
    if normalized == "promote" and all(_is_public_only_evidence(item) for item in references):
        raise ValueError("promotion requires non-public-score evidence")
    return {"decision": normalized, "evidence": references}


def amendment_payload(
    *, target_event_id: str, correction_reason: str, replacement_fields: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "target_event_id": _bounded_text(target_event_id, "target event ID", 160),
        "correction_reason": _bounded_text(correction_reason, "correction reason", 1000),
        "replacement_fields": normalize_exact_values(dict(replacement_fields)),
    }
