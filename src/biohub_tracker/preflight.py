from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .guard import parse_hours
from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file, utc_now
from .ledger import workspace_file


SCHEMA_VERSION = 1
BASE_CHECKS = (
    "imports",
    "inputs",
    "single_batch",
    "model_step",
    "checkpoint_roundtrip",
    "output_location",
)
LONG_RUN_CHECKS = ("dense_memory", "dataset_coverage")
PASSING_STATUSES = frozenset({"passed", "not_applicable"})


class PreflightError(ValueError):
    pass


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PreflightError("preflight created_at is malformed") from exc
    if result.tzinfo is None:
        raise PreflightError("preflight created_at must include a timezone")
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    evidence: tuple[Mapping[str, str], ...]
    detail: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        status: str,
        evidence_paths: Sequence[str | Path],
        workspace_root: Path,
        *,
        detail: str = "",
    ) -> "PreflightCheck":
        normalized = str(status).strip().casefold()
        if normalized not in {"passed", "failed", "not_applicable"}:
            raise PreflightError(f"invalid preflight status for {name}: {status}")
        records: list[dict[str, str]] = []
        for path in evidence_paths:
            candidate = workspace_file(workspace_root, path)
            records.append(
                {
                    "path": candidate.relative_to(workspace_root.resolve()).as_posix(),
                    "sha256": sha256_file(candidate),
                }
            )
        if normalized == "passed" and not records:
            raise PreflightError(f"passed check {name} requires hashed evidence")
        if normalized == "not_applicable" and not str(detail).strip():
            raise PreflightError(f"not_applicable check {name} requires a reason")
        return cls(
            name=str(name).strip(),
            status=normalized,
            evidence=tuple(records),
            detail=str(detail).strip()[:1000],
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreflightCheck":
        evidence = value.get("evidence")
        if not isinstance(evidence, list) or not all(isinstance(item, Mapping) for item in evidence):
            raise PreflightError("preflight check evidence must be a list of objects")
        return cls(
            name=str(value.get("name", "")).strip(),
            status=str(value.get("status", "")).strip().casefold(),
            evidence=tuple(dict(item) for item in evidence),
            detail=str(value.get("detail", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [dict(item) for item in self.evidence]
        return value


@dataclass(frozen=True)
class PreflightReport:
    run_id: str
    created_at: str
    checks: tuple[PreflightCheck, ...]
    schema_version: int = SCHEMA_VERSION
    report_sha256: str = ""

    @classmethod
    def create(
        cls,
        run_id: str,
        checks: Sequence[PreflightCheck],
        *,
        created_at: datetime | None = None,
    ) -> "PreflightReport":
        report = cls(
            run_id=str(run_id).strip(),
            created_at=_utc_text(created_at or utc_now()),
            checks=tuple(checks),
        )
        return cls(**{**report.__dict__, "report_sha256": report.computed_sha256()})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreflightReport":
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise PreflightError("unsupported preflight schema version")
        raw_checks = value.get("checks")
        if not isinstance(raw_checks, list):
            raise PreflightError("preflight checks must be a list")
        return cls(
            run_id=str(value.get("run_id", "")).strip(),
            created_at=str(value.get("created_at", "")).strip(),
            checks=tuple(PreflightCheck.from_dict(item) for item in raw_checks),
            schema_version=SCHEMA_VERSION,
            report_sha256=str(value.get("report_sha256", "")).strip().casefold(),
        )

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "checks": [check.to_dict() for check in self.checks],
        }

    def computed_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.unsigned_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "report_sha256": self.report_sha256}


def write_preflight_report(path: str | Path, report: PreflightReport) -> Path:
    if not secrets.compare_digest(report.report_sha256, report.computed_sha256()):
        raise PreflightError("preflight report hash is invalid")
    return atomic_write_json(path, report.to_dict())


def validate_preflight(
    report_path: str | Path,
    workspace_root: Path,
    *,
    run_id: str,
    declared_max_runtime_hours: Any,
    now: datetime | None = None,
    max_age_seconds: int = 3600,
) -> PreflightReport:
    import json

    path = workspace_file(workspace_root, report_path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise PreflightError("preflight report must be an object")
    report = PreflightReport.from_dict(value)
    if not report.run_id or report.run_id != run_id:
        raise PreflightError("preflight report belongs to a different run")
    if not secrets.compare_digest(report.report_sha256, report.computed_sha256()):
        raise PreflightError("preflight report content hash changed")
    created = _parse_time(report.created_at)
    current = (now or utc_now()).astimezone(timezone.utc)
    age = (current - created).total_seconds()
    if age < -60 or age > max_age_seconds:
        raise PreflightError("preflight report is stale or from the future")
    by_name: dict[str, PreflightCheck] = {}
    for check in report.checks:
        if not check.name or check.name in by_name:
            raise PreflightError("preflight check names must be nonempty and unique")
        if check.status not in PASSING_STATUSES:
            raise PreflightError(f"preflight check did not pass: {check.name}")
        if check.status == "not_applicable" and check.name != "model_step":
            raise PreflightError(f"only model_step may be not_applicable: {check.name}")
        if check.status == "not_applicable" and not check.detail:
            raise PreflightError("not_applicable model_step requires a reason")
        for evidence in check.evidence:
            evidence_path = evidence.get("path")
            expected = str(evidence.get("sha256", "")).casefold()
            candidate = workspace_file(workspace_root, str(evidence_path))
            if not expected or not secrets.compare_digest(sha256_file(candidate), expected):
                raise PreflightError(f"preflight evidence changed: {evidence_path}")
        if check.status == "passed" and not check.evidence:
            raise PreflightError(f"passed check lacks evidence: {check.name}")
        by_name[check.name] = check
    required = set(BASE_CHECKS)
    runtime = parse_hours(declared_max_runtime_hours, field="declared maximum runtime")
    if runtime > Decimal("1.00"):
        required.update(LONG_RUN_CHECKS)
    missing = sorted(required - by_name.keys())
    if missing:
        raise PreflightError(f"preflight report is missing required checks: {missing}")
    return report
