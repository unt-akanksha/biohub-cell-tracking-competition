from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol, Sequence

from .io import canonical_json_bytes, sha256_bytes, utc_now
from .ledger import RunStatus


TWO_PLACES = Decimal("0.01")
ACTIVE_STATUSES = frozenset({"RUNNING", "QUEUED", "PENDING"})
_HOURS = re.compile(r"^[+]?(?:\d+)(?:\.\d+)?(?:h)?$")


class GuardInputError(ValueError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code


class GuardRunner(Protocol):
    def run_json(self, args: Sequence[str]) -> Any: ...

    def kernel_status(self, ref: str) -> str: ...


def parse_hours(value: Any, *, field: str = "hours") -> Decimal:
    if isinstance(value, bool):
        raise GuardInputError("MALFORMED_HOURS", f"{field} is not a decimal hour value")
    text = str(value).strip()
    if not _HOURS.fullmatch(text):
        raise GuardInputError("MALFORMED_HOURS", f"{field} is not a decimal hour value")
    try:
        parsed = Decimal(text.removesuffix("h"))
    except InvalidOperation as exc:  # pragma: no cover - regex is the primary guard
        raise GuardInputError("MALFORMED_HOURS", f"{field} is not a decimal hour value") from exc
    if not parsed.is_finite():
        raise GuardInputError("MALFORMED_HOURS", f"{field} is not finite")
    return parsed


def hours_text(value: Decimal) -> str:
    return format(value.quantize(TWO_PLACES), ".2f")


def _parse_refresh(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise GuardInputError("QUOTA_REFRESH_UNAVAILABLE", "GPU quota refresh time is missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GuardInputError("QUOTA_REFRESH_UNAVAILABLE", "GPU quota refresh time is malformed") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class QuotaSnapshot:
    used: Decimal
    remaining: Decimal
    total: Decimal
    refresh_at: str
    captured_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "resource": "GPU",
            "used_hours": hours_text(self.used),
            "remaining_hours": hours_text(self.remaining),
            "total_hours": hours_text(self.total),
            "refresh_at": self.refresh_at,
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class ActiveKernel:
    ref: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GuardDecision:
    run_id: str
    authorized: bool
    reason_codes: tuple[str, ...]
    declared_max_runtime: Decimal
    reserve: Decimal
    notebook_maximum: Decimal
    remaining: Decimal | None
    projected_remaining: Decimal | None
    quota_snapshot_hash: str | None
    active_kernels: tuple[ActiveKernel, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "authorized": self.authorized,
            "reason_codes": list(self.reason_codes),
            "declared_max_runtime_hours": hours_text(self.declared_max_runtime),
            "reserve_hours": hours_text(self.reserve),
            "notebook_maximum_hours": hours_text(self.notebook_maximum),
            "remaining_hours": hours_text(self.remaining) if self.remaining is not None else None,
            "projected_remaining_hours": (
                hours_text(self.projected_remaining)
                if self.projected_remaining is not None
                else None
            ),
            "quota_snapshot_hash": self.quota_snapshot_hash,
            "active_kernels": [item.to_dict() for item in self.active_kernels],
        }


def read_gpu_quota(runner: GuardRunner) -> QuotaSnapshot:
    payload = runner.run_json(["quota", "--format", "json"])
    if not isinstance(payload, list):
        raise GuardInputError("QUOTA_UNAVAILABLE", "Kaggle quota payload is not a list")
    rows = [
        row
        for row in payload
        if isinstance(row, Mapping) and str(row.get("resource", "")).strip().upper() == "GPU"
    ]
    if len(rows) != 1:
        raise GuardInputError("GPU_QUOTA_ROW_AMBIGUOUS", "expected exactly one GPU quota row")
    row = rows[0]
    try:
        used = parse_hours(row.get("used"), field="GPU used")
        remaining = parse_hours(row.get("remaining"), field="GPU remaining")
        total = parse_hours(row.get("total"), field="GPU total")
    except GuardInputError as exc:
        raise GuardInputError("GPU_QUOTA_MALFORMED", str(exc)) from exc
    if min(used, remaining, total) < 0 or total <= 0 or used + remaining > total + TWO_PLACES:
        raise GuardInputError("GPU_QUOTA_MALFORMED", "GPU quota values are inconsistent")
    return QuotaSnapshot(
        used=used,
        remaining=remaining,
        total=total,
        refresh_at=_parse_refresh(row.get("refreshAt")),
        captured_at=utc_now().isoformat().replace("+00:00", "Z"),
    )


def list_active_gpu_kernels(runner: GuardRunner, competition_slug: str) -> list[ActiveKernel]:
    payload = runner.run_json(
        [
            "kernels",
            "list",
            "--mine",
            "--competition",
            competition_slug,
            "--format",
            "json",
            "--sort-by",
            "dateRun",
            "--page-size",
            "20",
        ]
    )
    if not isinstance(payload, list):
        raise GuardInputError("KERNEL_STATUS_UNAVAILABLE", "Kaggle kernels payload is not a list")
    active: list[ActiveKernel] = []
    seen: set[str] = set()
    refs: list[str] = []
    for row in payload:
        if not isinstance(row, Mapping) or not str(row.get("ref", "")).strip():
            raise GuardInputError("KERNEL_STATUS_UNAVAILABLE", "kernel listing has a missing ref")
        ref = str(row["ref"]).strip()
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)

    # An executing or newly queued version sorts ahead of historical completed
    # versions by dateRun. Inspecting the newest 20 sequentially avoids Kaggle's
    # status-endpoint rate limit while covering far more than its concurrency cap.
    for ref in refs:
        try:
            status = str(runner.kernel_status(ref)).removeprefix("KernelWorkerStatus.").upper()
        except Exception as exc:
            raise GuardInputError(
                "KERNEL_STATUS_UNAVAILABLE", f"could not resolve status for {ref}"
            ) from exc
        if status in ACTIVE_STATUSES:
            active.append(ActiveKernel(ref=ref, status=status))
    return sorted(active, key=lambda item: item.ref)


def snapshot_hash(quota: QuotaSnapshot, active_kernels: Sequence[ActiveKernel]) -> str:
    # captured_at is evidence metadata, not state; excluding it permits an immediate unchanged-state recheck.
    quota_state = quota.to_dict()
    quota_state.pop("captured_at")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "quota": quota_state,
                "active_kernels": [item.to_dict() for item in sorted(active_kernels, key=lambda x: x.ref)],
            }
        )
    )


def evaluate_guard(
    *,
    run_id: str,
    registered_status: RunStatus | str | None,
    registered_declared_runtime: Any | None = None,
    declared_max_runtime: Any,
    quota: QuotaSnapshot | None,
    active_kernels: Sequence[ActiveKernel],
    reserve: Any = "8.00",
    notebook_maximum: Any = "12.00",
    input_error_code: str | None = None,
) -> GuardDecision:
    runtime = parse_hours(declared_max_runtime, field="declared maximum runtime")
    reserve_hours = parse_hours(reserve, field="GPU reserve")
    maximum = parse_hours(notebook_maximum, field="notebook maximum")
    reasons: list[str] = []
    if registered_status is None:
        reasons.append("RUN_NOT_REGISTERED")
    elif RunStatus(registered_status) is not RunStatus.REGISTERED:
        reasons.append("RUN_NOT_LAUNCHABLE")
    if registered_declared_runtime is not None:
        registered_runtime = parse_hours(
            registered_declared_runtime, field="registered declared maximum runtime"
        )
        if runtime != registered_runtime:
            reasons.append("DECLARED_RUNTIME_MISMATCH")
    if runtime <= 0:
        reasons.append("RUNTIME_NOT_POSITIVE")
    elif runtime > maximum:
        reasons.append("RUNTIME_ABOVE_NOTEBOOK_LIMIT")
    if input_error_code:
        reasons.append(input_error_code)
    if active_kernels:
        reasons.append("ACTIVE_GPU_KERNEL")
    remaining = quota.remaining if quota else None
    projected = remaining - runtime if remaining is not None else None
    if quota is None and not input_error_code:
        reasons.append("QUOTA_UNAVAILABLE")
    elif projected is not None and projected < reserve_hours:
        reasons.append("GPU_RESERVE_VIOLATION")
    return GuardDecision(
        run_id=run_id,
        authorized=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        declared_max_runtime=runtime,
        reserve=reserve_hours,
        notebook_maximum=maximum,
        remaining=remaining,
        projected_remaining=projected,
        quota_snapshot_hash=snapshot_hash(quota, active_kernels) if quota else None,
        active_kernels=tuple(active_kernels),
    )
