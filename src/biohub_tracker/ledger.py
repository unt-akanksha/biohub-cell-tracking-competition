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
    EXACT_EVALUATION_REGISTERED = "exact_evaluation_registered"
    EXACT_EVALUATION_STARTED = "exact_evaluation_started"
    EXACT_EVALUATION_MATERIALIZED = "exact_evaluation_materialized"
    EXACT_EVALUATION_COMPLETED = "exact_evaluation_completed"
    EXACT_EVALUATION_FAILED = "exact_evaluation_failed"
    EXACT_PROMOTION_DECISION = "exact_promotion_decision"
    EXACT_PROMOTION_EXCEPTION = "exact_promotion_exception"
    CPU_ACCEPTANCE_REGISTERED = "cpu_acceptance_registered"
    CPU_ACCEPTANCE_STARTED = "cpu_acceptance_started"
    CPU_ACCEPTANCE_INPUTS_BOUND = "cpu_acceptance_inputs_bound"
    CPU_ACCEPTANCE_COMPLETED = "cpu_acceptance_completed"
    CPU_ACCEPTANCE_FAILED = "cpu_acceptance_failed"


class RunStatus(StrEnum):
    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class ExactEvaluationStatus(StrEnum):
    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CpuAcceptanceStatus(StrEnum):
    REGISTERED = "registered"
    RUNNING = "running"
    INPUTS_BOUND = "inputs_bound"
    COMPLETED = "completed"
    FAILED = "failed"


EXACT_MEMBER_FIELDS = (
    "role",
    "fold_id",
    "producer_run_id",
    "producer_registration_event_sha256",
    "producer_terminal_event_sha256",
    *(name for name in PRODUCER_REGISTRATION_FIELDS if name != "fold_id"),
    "graph_inventory_sha256",
    "artifact_hashes",
)
CPU_EXACT_MEMBER_FIELD = "producer_input_binding_event_sha256"
REVIEW_EXCEPTION_AUTHORIZATIONS = frozenset(
    {"diagnostic-only", "phase3-experiment-only", "phase4-experiment-only"}
)

_EXACT_EVENT_TYPES = {
    EventType.EXACT_EVALUATION_REGISTERED,
    EventType.EXACT_EVALUATION_STARTED,
    EventType.EXACT_EVALUATION_MATERIALIZED,
    EventType.EXACT_EVALUATION_COMPLETED,
    EventType.EXACT_EVALUATION_FAILED,
}
_EXACT_DECISION_EVENT_TYPES = {
    EventType.EXACT_PROMOTION_DECISION,
    EventType.EXACT_PROMOTION_EXCEPTION,
}
_CPU_EVENT_TYPES = {
    EventType.CPU_ACCEPTANCE_REGISTERED,
    EventType.CPU_ACCEPTANCE_STARTED,
    EventType.CPU_ACCEPTANCE_INPUTS_BOUND,
    EventType.CPU_ACCEPTANCE_COMPLETED,
    EventType.CPU_ACCEPTANCE_FAILED,
}
_EXPERIMENT_EVENT_TYPES = (
    set(EventType) - _EXACT_EVENT_TYPES - _EXACT_DECISION_EVENT_TYPES - _CPU_EVENT_TYPES
)


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
        if set(value) != required or not isinstance(value.get("payload"), Mapping):
            raise ValueError("event has unknown or missing fields")
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


@dataclass
class ExactEvaluationState:
    evaluation_run_id: str
    status: ExactEvaluationStatus
    registered: dict[str, Any]
    events: list[ExperimentEvent] = field(default_factory=list)
    materialized: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    exceptions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CpuAcceptanceState:
    run_id: str
    status: CpuAcceptanceStatus
    registered: dict[str, Any]
    events: list[ExperimentEvent] = field(default_factory=list)
    started: dict[str, Any] | None = None
    inputs_bound: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None


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
        EventType.EXACT_EVALUATION_REGISTERED: 0,
        EventType.EXACT_EVALUATION_STARTED: 2,
        EventType.EXACT_EVALUATION_MATERIALIZED: 3,
        EventType.EXACT_EVALUATION_COMPLETED: 4,
        EventType.EXACT_EVALUATION_FAILED: 4,
        EventType.EXACT_PROMOTION_DECISION: 4,
        EventType.EXACT_PROMOTION_EXCEPTION: 5,
        EventType.CPU_ACCEPTANCE_REGISTERED: 0,
        EventType.CPU_ACCEPTANCE_STARTED: 1,
        EventType.CPU_ACCEPTANCE_INPUTS_BOUND: 2,
        EventType.CPU_ACCEPTANCE_COMPLETED: 3,
        EventType.CPU_ACCEPTANCE_FAILED: 3,
    }
    return (_parse_time(event.created_at), precedence[event.event_type], event.event_id)


def reconstruct_runs(events: Iterable[ExperimentEvent]) -> dict[str, RunState]:
    grouped: dict[str, list[ExperimentEvent]] = {}
    for event in events:
        if event.event_type not in _EXPERIMENT_EVENT_TYPES:
            continue
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


def event_sha256(event: ExperimentEvent) -> str:
    return sha256_bytes(canonical_json_bytes(event.to_dict()))


def reconstruct_cpu_acceptances(
    events: Iterable[ExperimentEvent],
) -> dict[str, CpuAcceptanceState]:
    grouped: dict[str, list[ExperimentEvent]] = {}
    for event in events:
        if event.event_type in _CPU_EVENT_TYPES:
            grouped.setdefault(event.run_id, []).append(event)
    result: dict[str, CpuAcceptanceState] = {}
    for run_id in sorted(grouped):
        ordered = sorted(grouped[run_id], key=_event_order)
        registered = [
            item for item in ordered if item.event_type is EventType.CPU_ACCEPTANCE_REGISTERED
        ]
        if len(registered) != 1:
            raise TransitionError(f"CPU acceptance {run_id} must have exactly one registered event")
        state = CpuAcceptanceState(
            run_id=run_id,
            status=CpuAcceptanceStatus.REGISTERED,
            registered=dict(registered[0].payload),
            events=ordered,
        )
        for item in ordered:
            if item.event_type is EventType.CPU_ACCEPTANCE_REGISTERED:
                continue
            if item.event_type is EventType.CPU_ACCEPTANCE_STARTED:
                if state.status is not CpuAcceptanceStatus.REGISTERED:
                    raise TransitionError(f"CPU acceptance {run_id} has an illegal start")
                state.status = CpuAcceptanceStatus.RUNNING
                state.started = dict(item.payload)
            elif item.event_type is EventType.CPU_ACCEPTANCE_INPUTS_BOUND:
                if state.status is not CpuAcceptanceStatus.RUNNING:
                    raise TransitionError(f"CPU acceptance {run_id} has an illegal input binding")
                state.status = CpuAcceptanceStatus.INPUTS_BOUND
                state.inputs_bound = dict(item.payload)
            elif item.event_type is EventType.CPU_ACCEPTANCE_COMPLETED:
                if state.status is not CpuAcceptanceStatus.INPUTS_BOUND:
                    raise TransitionError(f"CPU acceptance {run_id} has an illegal completion")
                state.status = CpuAcceptanceStatus.COMPLETED
                state.terminal = dict(item.payload)
            elif item.event_type is EventType.CPU_ACCEPTANCE_FAILED:
                if state.status in {CpuAcceptanceStatus.COMPLETED, CpuAcceptanceStatus.FAILED}:
                    raise TransitionError(f"CPU acceptance {run_id} has a late failure")
                state.status = CpuAcceptanceStatus.FAILED
                state.terminal = dict(item.payload)
        result[run_id] = state
    return result


def _exact_member_sort_key(value: Mapping[str, Any]) -> tuple[str, str]:
    return (str(value.get("role", "")), str(value.get("fold_id", "")))


def _normalize_exact_member(value: Any) -> dict[str, Any]:
    fields = set(value) if isinstance(value, Mapping) else set()
    if not isinstance(value, Mapping) or fields not in (
        set(EXACT_MEMBER_FIELDS),
        {*EXACT_MEMBER_FIELDS, CPU_EXACT_MEMBER_FIELD},
    ):
        raise ValueError(f"exact evaluation member requires exactly {list(EXACT_MEMBER_FIELDS)}")
    role = _bounded_text(value["role"], "member role", 20)
    if role not in {"baseline", "candidate"}:
        raise ValueError(f"invalid exact evaluation member role: {role}")
    result: dict[str, Any] = {
        "role": role,
        "fold_id": _bounded_text(value["fold_id"], "member fold_id", 200),
        "producer_run_id": _bounded_text(
            value["producer_run_id"], "member producer_run_id", 160
        ),
        "producer_registration_event_sha256": _sha256_text(
            value["producer_registration_event_sha256"],
            "producer_registration_event_sha256",
        ),
        "producer_terminal_event_sha256": _sha256_text(
            value["producer_terminal_event_sha256"], "producer_terminal_event_sha256"
        ),
    }
    for name in PRODUCER_REGISTRATION_FIELDS:
        result[name] = (
            _bounded_text(value[name], name, 200)
            if name == "fold_id"
            else _sha256_text(value[name], name)
        )
    result["graph_inventory_sha256"] = _sha256_text(
        value["graph_inventory_sha256"], "graph_inventory_sha256"
    )
    result["artifact_hashes"] = _artifact_hashes(value["artifact_hashes"])
    names = list(EXACT_MEMBER_FIELDS)
    if CPU_EXACT_MEMBER_FIELD in value:
        result[CPU_EXACT_MEMBER_FIELD] = _sha256_text(
            value[CPU_EXACT_MEMBER_FIELD], CPU_EXACT_MEMBER_FIELD
        )
        names.insert(4, CPU_EXACT_MEMBER_FIELD)
    return {name: result[name] for name in names}


def _normalize_exact_members(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("exact evaluation requires four role/fold members")
    members = [_normalize_exact_member(item) for item in value]
    if members != sorted(members, key=_exact_member_sort_key):
        raise ValueError("exact evaluation members must be in canonical role/fold order")
    slots = [(item["role"], item["fold_id"]) for item in members]
    if len(set(slots)) != 4:
        raise ValueError("exact evaluation role/fold member slots must be unique")
    folds = sorted({item["fold_id"] for item in members})
    if len(folds) != 2 or set(slots) != {
        (role, fold) for role in ("baseline", "candidate") for fold in folds
    }:
        raise ValueError("exact evaluation requires baseline and candidate for both folds")
    return members


def resolved_exact_member(
    events: Sequence[ExperimentEvent], *, role: str, fold_id: str, producer_run_id: str
) -> dict[str, Any]:
    runs = reconstruct_runs(events)
    state = runs.get(producer_run_id)
    if state is not None:
        if state.status is not RunStatus.COMPLETED or state.terminal is None:
            raise TransitionError(f"nonterminal exact-evaluation producer: {producer_run_id}")
        try:
            registered = validate_producer_registration_evidence(state.registered, required=True)
            terminal = validate_producer_terminal_evidence(state.terminal, required=True)
        except ValueError as exc:
            raise TransitionError(f"ineligible exact-evaluation producer: {producer_run_id}") from exc
        if registered["fold_id"] != fold_id:
            raise TransitionError(
                f"exact-evaluation producer {producer_run_id} belongs to {registered['fold_id']}, not {fold_id}"
            )
        registered_event = next(
            item for item in state.events if item.event_type is EventType.REGISTERED
        )
        terminal_event = next(
            item for item in state.events if item.event_type is EventType.COMPLETED
        )
        return _normalize_exact_member(
            {
                "role": role,
                "fold_id": fold_id,
                "producer_run_id": producer_run_id,
                "producer_registration_event_sha256": event_sha256(registered_event),
                "producer_terminal_event_sha256": event_sha256(terminal_event),
                **registered,
                "graph_inventory_sha256": terminal["graph_inventory_sha256"],
                "artifact_hashes": terminal["artifact_hashes"],
            }
        )
    controls = reconstruct_cpu_acceptances(events)
    control = controls.get(producer_run_id)
    if control is None:
        raise TransitionError(f"unknown exact-evaluation producer: {producer_run_id}")
    if control.status is not CpuAcceptanceStatus.COMPLETED or control.terminal is None:
        raise TransitionError(f"nonterminal exact-evaluation producer: {producer_run_id}")
    if control.inputs_bound is None:
        raise TransitionError(f"ineligible exact-evaluation producer: {producer_run_id}")
    fold = next(
        (item for item in control.inputs_bound["folds"] if item["fold_id"] == fold_id), None
    )
    if fold is None:
        raise TransitionError(
            f"exact-evaluation producer {producer_run_id} has no bound fold {fold_id}"
        )
    registered_event = next(
        item for item in control.events if item.event_type is EventType.CPU_ACCEPTANCE_REGISTERED
    )
    binding_event = next(
        item for item in control.events if item.event_type is EventType.CPU_ACCEPTANCE_INPUTS_BOUND
    )
    terminal_event = next(
        item for item in control.events if item.event_type is EventType.CPU_ACCEPTANCE_COMPLETED
    )
    return _normalize_exact_member(
        {
            "role": role,
            "fold_id": fold_id,
            "producer_run_id": producer_run_id,
            "producer_registration_event_sha256": event_sha256(registered_event),
            CPU_EXACT_MEMBER_FIELD: event_sha256(binding_event),
            "producer_terminal_event_sha256": event_sha256(terminal_event),
            "manifest_sha256": control.inputs_bound["manifest_sha256"],
            "train_membership_sha256": fold["train_membership_sha256"],
            "calibration_membership_sha256": fold["calibration_membership_sha256"],
            "evaluation_membership_sha256": fold["evaluation_membership_sha256"],
            "model_sha256": control.registered["control_model_sha256"],
            "config_sha256": control.registered["config_sha256"],
            "code_sha256": control.registered["code_sha256"],
            "data_sha256": control.registered["data_source_sha256"],
            "graph_inventory_sha256": control.terminal["graph_inventory_sha256"],
            "artifact_hashes": control.terminal["artifact_hashes"],
        }
    )


def _validate_registered_exact_members(
    events: Sequence[ExperimentEvent], members: Sequence[Mapping[str, Any]]
) -> None:
    for member in members:
        expected = resolved_exact_member(
            events,
            role=str(member["role"]),
            fold_id=str(member["fold_id"]),
            producer_run_id=str(member["producer_run_id"]),
        )
        if canonical_json_bytes(expected) != canonical_json_bytes(member):
            raise TransitionError(
                f"exact-evaluation producer evidence mismatch: {member['role']}/{member['fold_id']}"
            )


def reconstruct_exact_evaluations(
    events: Iterable[ExperimentEvent],
) -> dict[str, ExactEvaluationState]:
    materialized = list(events)
    grouped: dict[str, list[ExperimentEvent]] = {}
    for event in materialized:
        if event.event_type in _EXACT_EVENT_TYPES:
            grouped.setdefault(event.run_id, []).append(event)
    result: dict[str, ExactEvaluationState] = {}
    for evaluation_run_id in sorted(grouped):
        ordered = sorted(grouped[evaluation_run_id], key=_event_order)
        registered = [
            event
            for event in ordered
            if event.event_type is EventType.EXACT_EVALUATION_REGISTERED
        ]
        if len(registered) != 1:
            raise TransitionError(
                f"exact evaluation {evaluation_run_id} must have exactly one registered event"
            )
        state = ExactEvaluationState(
            evaluation_run_id=evaluation_run_id,
            status=ExactEvaluationStatus.REGISTERED,
            registered=dict(registered[0].payload),
            events=ordered,
        )
        for event in ordered:
            if event.event_type is EventType.EXACT_EVALUATION_REGISTERED:
                continue
            if event.event_type is EventType.EXACT_EVALUATION_STARTED:
                if state.status is not ExactEvaluationStatus.REGISTERED:
                    raise TransitionError(
                        f"exact evaluation {evaluation_run_id} has an illegal repeated/late start"
                    )
                state.status = ExactEvaluationStatus.RUNNING
            elif event.event_type is EventType.EXACT_EVALUATION_MATERIALIZED:
                if (
                    state.status is not ExactEvaluationStatus.RUNNING
                    or state.materialized is not None
                ):
                    raise TransitionError(
                        f"exact evaluation {evaluation_run_id} has an illegal materialization"
                    )
                state.materialized = dict(event.payload)
            elif event.event_type in {
                EventType.EXACT_EVALUATION_COMPLETED,
                EventType.EXACT_EVALUATION_FAILED,
            }:
                if state.status is not ExactEvaluationStatus.RUNNING:
                    raise TransitionError(
                        f"exact evaluation {evaluation_run_id} has an illegal terminal event"
                    )
                state.status = ExactEvaluationStatus(
                    "completed"
                    if event.event_type is EventType.EXACT_EVALUATION_COMPLETED
                    else "failed"
                )
                state.terminal = dict(event.payload)
        result[evaluation_run_id] = state
    for event in sorted(
        (item for item in materialized if item.event_type in _EXACT_DECISION_EVENT_TYPES),
        key=_event_order,
    ):
        state = result.get(event.run_id)
        if state is None:
            raise TransitionError(f"decision for unknown exact evaluation {event.run_id}")
        state.events.append(event)
        if event.event_type is EventType.EXACT_PROMOTION_DECISION:
            if state.decision is not None:
                raise TransitionError(f"exact evaluation {event.run_id} has duplicate decisions")
            state.decision = dict(event.payload)
        else:
            if state.decision is None or state.decision.get("state") != "review_required":
                raise TransitionError(f"exact evaluation {event.run_id} has an illegal exception")
            state.exceptions.append(dict(event.payload))
    return result


def validate_transition(events: Sequence[ExperimentEvent], event: ExperimentEvent) -> None:
    if any(existing.event_id == event.event_id for existing in events):
        raise TransitionError(f"duplicate event ID: {event.event_id}")
    if event.event_type in _EXACT_EVENT_TYPES:
        _validate_exact_transition(events, event)
        return
    if event.event_type in _CPU_EVENT_TYPES:
        _validate_cpu_transition(events, event)
        return
    if event.event_type in _EXACT_DECISION_EVENT_TYPES:
        _validate_exact_decision_transition(events, event)
        return
    runs = reconstruct_runs(events) if events else {}
    existing = runs.get(event.run_id)
    if event.event_type is EventType.REGISTERED:
        evaluations = reconstruct_exact_evaluations(events)
        controls = reconstruct_cpu_acceptances(events)
        reserved_evaluation_ids = {
            str(state.registered["evaluation_run_id"]) for state in controls.values()
        }
        if (
            existing
            or event.run_id in evaluations
            or event.run_id in controls
            or event.run_id in reserved_evaluation_ids
        ):
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
        self.owner_token: str | None = None

    @staticmethod
    def _process_start_identity(pid: int) -> str | None:
        if pid <= 0:
            return None
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                process_query_limited_information, False, pid
            )
            if not handle:
                error = ctypes.windll.kernel32.GetLastError()  # type: ignore[attr-defined]
                return None if error == 87 else f"active-unobservable:{pid}"
            try:
                creation = ctypes.c_ulonglong()
                exit_time = ctypes.c_ulonglong()
                kernel = ctypes.c_ulonglong()
                user = ctypes.c_ulonglong()
                if not ctypes.windll.kernel32.GetProcessTimes(  # type: ignore[attr-defined]
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return f"active-unobservable:{pid}"
                return f"windows-filetime:{creation.value}"
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        proc_stat = Path(f"/proc/{pid}/stat")
        try:
            fields = proc_stat.read_text(encoding="ascii").split()
            return f"proc-start-ticks:{fields[21]}"
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, IndexError):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            except (PermissionError, OSError):
                return f"active-unobservable:{pid}"
            return f"active-unobservable:{pid}"

    def _metadata(self) -> dict[str, Any]:
        now = time.time()
        return {
            "schema_version": "biohub.ledger-lock.v1",
            "pid": os.getpid(),
            "process_start": self._process_start_identity(os.getpid()),
            "created_at": datetime.fromtimestamp(now, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "created_at_epoch_seconds": now,
            "owner_token": secrets.token_hex(32),
        }

    def _recover_stale(self) -> bool:
        # A pathname cannot be removed with compare-and-swap semantics on every
        # supported platform.  Automatic stale recovery could therefore rename a
        # new owner's live lock after inspecting an older file.  Fail closed: an
        # operator must verify that no writer is active and remove a stale lock
        # explicitly before retrying.
        return False

    def __enter__(self) -> "_ExclusiveLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                metadata = self._metadata()
                self.owner_token = str(metadata["owner_token"])
                os.write(self.descriptor, canonical_json_bytes(metadata) + b"\n")
                os.fsync(self.descriptor)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    if self._recover_stale():
                        deadline = time.monotonic() + self.timeout_seconds
                        continue
                    raise LedgerLockTimeout(f"timed out waiting for ledger lock: {self.path}")
                time.sleep(0.01)

    def __exit__(self, *_: Any) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        try:
            metadata = json.loads(self.path.read_bytes())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if (
            isinstance(metadata, dict)
            and self.owner_token is not None
            and secrets.compare_digest(str(metadata.get("owner_token")), self.owner_token)
        ):
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
        reconstruct_cpu_acceptances(events)
        reconstruct_exact_evaluations(events)
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
            lifecycle_events = [item for item in events if item.run_id == event.run_id]
            if lifecycle_events and _parse_time(event.created_at) < max(
                _parse_time(item.created_at) for item in lifecycle_events
            ):
                raise TransitionError("event created_at precedes its lifecycle history")
            validate_transition(events, event)
            payload = canonical_json_bytes(event.to_dict()) + b"\n"
            # Validate the exact durable candidate, including reconstruction order,
            # before writing a single byte. Caller-controlled timestamps must never
            # be able to append an event that makes the ledger unreadable.
            self._decode(existing_bytes + payload, quarantine=False)
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


def _owned_slug(value: Any, name: str) -> str:
    text = _bounded_text(value, name, 160)
    if re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", text) is None:
        raise ValueError(f"{name} must be an owned Kaggle owner/slug")
    return text


def cpu_acceptance_registration_payload(
    *,
    run_id: str,
    purpose: str,
    request_nonce: str,
    acceptance_request_sha256: str,
    evaluation_run_id: str,
    kernel_slug: str,
    runtime_dataset_slug: str,
    runtime_bundle_name: str,
    runtime_bundle_sha256: str,
    runtime_bundle_inventory_sha256: str,
    runtime_bundle_uncompressed_size_bytes: Any,
    runtime_bundle_file_count: Any,
    scorer_lock_sha256: str,
    environment_lock_sha256: str,
    manifest_policy_sha256: str,
    control_model_sha256: str,
    config_sha256: str,
    code_sha256: str,
    data_source_sha256: str,
    cpu_watchdog_minutes: Any,
) -> dict[str, Any]:
    try:
        watchdog = int(cpu_watchdog_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError("cpu_watchdog_minutes must be an integer") from exc
    if watchdog < 1 or watchdog >= 720:
        raise ValueError("cpu_watchdog_minutes must be between 1 and 719")
    try:
        bundle_size = int(runtime_bundle_uncompressed_size_bytes)
        bundle_count = int(runtime_bundle_file_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime bundle size/count must be integers") from exc
    if bundle_size < 1 or bundle_size > 2_147_483_648 or bundle_count < 1:
        raise ValueError("runtime bundle size/count is outside frozen bounds")
    bundle_name = _bounded_text(runtime_bundle_name, "runtime_bundle_name", 160)
    if bundle_name != "biohub-runtime-v1.biohubbundle":
        raise ValueError("runtime_bundle_name is not the canonical opaque bundle")
    nonce = _sha256_text(request_nonce, "request_nonce")
    return {
        "run_id": _bounded_text(run_id, "run_id", 160),
        "purpose": _bounded_text(purpose, "purpose", 160),
        "request_nonce": nonce,
        "acceptance_request_sha256": _sha256_text(
            acceptance_request_sha256, "acceptance_request_sha256"
        ),
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160),
        "kernel_slug": _owned_slug(kernel_slug, "kernel_slug"),
        "runtime_dataset_slug": _owned_slug(
            runtime_dataset_slug, "runtime_dataset_slug"
        ),
        "runtime_bundle_name": bundle_name,
        "runtime_bundle_sha256": _sha256_text(
            runtime_bundle_sha256, "runtime_bundle_sha256"
        ),
        "runtime_bundle_inventory_sha256": _sha256_text(
            runtime_bundle_inventory_sha256, "runtime_bundle_inventory_sha256"
        ),
        "runtime_bundle_uncompressed_size_bytes": bundle_size,
        "runtime_bundle_file_count": bundle_count,
        "scorer_lock_sha256": _sha256_text(scorer_lock_sha256, "scorer_lock_sha256"),
        "environment_lock_sha256": _sha256_text(
            environment_lock_sha256, "environment_lock_sha256"
        ),
        "manifest_policy_sha256": _sha256_text(
            manifest_policy_sha256, "manifest_policy_sha256"
        ),
        "control_model_sha256": _sha256_text(
            control_model_sha256, "control_model_sha256"
        ),
        "config_sha256": _sha256_text(config_sha256, "config_sha256"),
        "code_sha256": _sha256_text(code_sha256, "code_sha256"),
        "data_source_sha256": _sha256_text(data_source_sha256, "data_source_sha256"),
        "cpu_watchdog_minutes": watchdog,
        "accelerator": "none",
        "internet_enabled": False,
        "competition_submission_allowed": False,
    }


def cpu_acceptance_started_payload(
    *,
    run_id: str,
    registration_event_sha256: str,
    kernel_ref: str,
    runtime_dataset_ref: str,
) -> dict[str, Any]:
    return {
        "run_id": _bounded_text(run_id, "run_id", 160),
        "registration_event_sha256": _sha256_text(
            registration_event_sha256, "registration_event_sha256"
        ),
        "kernel_ref": _bounded_text(kernel_ref, "kernel_ref", 240),
        "runtime_dataset_ref": _bounded_text(
            runtime_dataset_ref, "runtime_dataset_ref", 240
        ),
        "accelerator": "none",
    }


def _normalize_cpu_folds(value: Any) -> list[dict[str, str]]:
    required = {
        "fold_id",
        "train_membership_sha256",
        "calibration_membership_sha256",
        "evaluation_membership_sha256",
    }
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("CPU input binding requires exactly two folds")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("CPU input-binding fold fields are invalid")
        result.append(
            {
                "fold_id": _bounded_text(item["fold_id"], "fold_id", 200),
                "train_membership_sha256": _sha256_text(
                    item["train_membership_sha256"], "train_membership_sha256"
                ),
                "calibration_membership_sha256": _sha256_text(
                    item["calibration_membership_sha256"],
                    "calibration_membership_sha256",
                ),
                "evaluation_membership_sha256": _sha256_text(
                    item["evaluation_membership_sha256"],
                    "evaluation_membership_sha256",
                ),
            }
        )
    result.sort(key=lambda item: item["fold_id"])
    if len({item["fold_id"] for item in result}) != 2:
        raise ValueError("CPU input-binding folds must be unique")
    return result


def cpu_acceptance_inputs_bound_payload(
    *, run_id: str, manifest_sha256: str, folds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "run_id": _bounded_text(run_id, "run_id", 160),
        "manifest_sha256": _sha256_text(manifest_sha256, "manifest_sha256"),
        "folds": _normalize_cpu_folds([dict(item) for item in folds]),
    }


def cpu_acceptance_completed_payload(
    *,
    run_id: str,
    actual_cpu_runtime_seconds: Any,
    peak_memory_mb: Any,
    remote_job_identity: str,
    graph_inventory_sha256: str,
    artifact_hashes: Mapping[str, Any],
    output_inventory_sha256: str,
    pending_payload_sha256: str,
    pending_envelope_sha256: str,
    reconciliation_sha256: str,
) -> dict[str, Any]:
    return {
        "run_id": _bounded_text(run_id, "run_id", 160),
        "actual_cpu_runtime_seconds": decimal_text(
            actual_cpu_runtime_seconds, "actual CPU runtime", positive=True
        ),
        "peak_memory_mb": decimal_text(peak_memory_mb, "peak memory", positive=True),
        "remote_job_identity": _bounded_text(
            remote_job_identity, "remote_job_identity", 240
        ),
        "graph_inventory_sha256": _sha256_text(
            graph_inventory_sha256, "graph_inventory_sha256"
        ),
        "artifact_hashes": _artifact_hashes(artifact_hashes),
        "output_inventory_sha256": _sha256_text(
            output_inventory_sha256, "output_inventory_sha256"
        ),
        "pending_payload_sha256": _sha256_text(
            pending_payload_sha256, "pending_payload_sha256"
        ),
        "pending_envelope_sha256": _sha256_text(
            pending_envelope_sha256, "pending_envelope_sha256"
        ),
        "reconciliation_sha256": _sha256_text(
            reconciliation_sha256, "reconciliation_sha256"
        ),
        "evidence_eligible": True,
        "promotion_eligible": False,
        "accelerator": "none",
        "competition_submission_performed": False,
    }


def cpu_acceptance_failed_payload(
    *,
    run_id: str,
    reason_code: str,
    detail: str,
    observed_artifact_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": _bounded_text(run_id, "run_id", 160),
        "reason_code": _bounded_text(reason_code, "reason_code", 160),
        "detail": _bounded_text(detail, "failure detail", 2000),
        "observed_artifact_hashes": (
            _artifact_hashes(observed_artifact_hashes)
            if observed_artifact_hashes
            else {}
        ),
        "evidence_eligible": False,
        "promotion_eligible": False,
        "accelerator": "none",
        "competition_submission_performed": False,
    }


def exact_evaluation_registration_payload(
    *,
    evaluation_run_id: str,
    scorer_lock_sha256: str,
    environment_lock_sha256: str,
    manifest_sha256: str,
    evaluation_policy_sha256: str,
    evidence_kind: str,
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    kind = _bounded_text(evidence_kind, "evidence_kind", 80)
    if kind not in {"synthetic_fixture", "official_data_control", "model_candidate"}:
        raise ValueError(f"invalid exact evaluation evidence kind: {kind}")
    return {
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160),
        "scorer_lock_sha256": _sha256_text(scorer_lock_sha256, "scorer_lock_sha256"),
        "environment_lock_sha256": _sha256_text(
            environment_lock_sha256, "environment_lock_sha256"
        ),
        "manifest_sha256": _sha256_text(manifest_sha256, "manifest_sha256"),
        "evaluation_policy_sha256": _sha256_text(
            evaluation_policy_sha256, "evaluation_policy_sha256"
        ),
        "evidence_kind": kind,
        "members": _normalize_exact_members([dict(item) for item in members]),
    }


def exact_evaluation_started_payload(*, evaluation_run_id: str) -> dict[str, str]:
    return {
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160)
    }


def _normalize_authoritative_inventories(value: Any) -> list[dict[str, str]]:
    required = {
        "role",
        "fold_id",
        "producer_run_id",
        "submission_graph_inventory_sha256",
        "roundtrip_evidence_sha256",
        "csv_sha256",
    }
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("completion requires four authoritative inventories")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("authoritative inventory fields are invalid")
        role = _bounded_text(item["role"], "inventory role", 20)
        if role not in {"baseline", "candidate"}:
            raise ValueError(f"invalid authoritative inventory role: {role}")
        result.append(
            {
                "role": role,
                "fold_id": _bounded_text(item["fold_id"], "inventory fold_id", 200),
                "producer_run_id": _bounded_text(
                    item["producer_run_id"], "inventory producer_run_id", 160
                ),
                "submission_graph_inventory_sha256": _sha256_text(
                    item["submission_graph_inventory_sha256"],
                    "submission_graph_inventory_sha256",
                ),
                "roundtrip_evidence_sha256": _sha256_text(
                    item["roundtrip_evidence_sha256"], "roundtrip_evidence_sha256"
                ),
                "csv_sha256": _sha256_text(item["csv_sha256"], "csv_sha256"),
            }
        )
    result.sort(key=_exact_member_sort_key)
    if len({(item["role"], item["fold_id"]) for item in result}) != 4:
        raise ValueError("authoritative inventory slots must be unique")
    return result


def exact_evaluation_completed_payload(
    *,
    evaluation_run_id: str,
    members: Sequence[Mapping[str, Any]],
    report_core_sha256: str,
    envelope_sha256: str,
    artifact_hashes: Mapping[str, Any],
    authoritative_inventories: Sequence[Mapping[str, Any]],
    promotion_eligible: bool = True,
    materialization_event_sha256: str | None = None,
) -> dict[str, Any]:
    result = {
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160),
        "members": _normalize_exact_members([dict(item) for item in members]),
        "report_core_sha256": _sha256_text(report_core_sha256, "report_core_sha256"),
        "envelope_sha256": _sha256_text(envelope_sha256, "envelope_sha256"),
        "artifact_hashes": _artifact_hashes(artifact_hashes),
        "authoritative_inventories": _normalize_authoritative_inventories(
            [dict(item) for item in authoritative_inventories]
        ),
        "promotion_eligible": bool(promotion_eligible),
    }
    if materialization_event_sha256 is not None:
        result["materialization_event_sha256"] = _sha256_text(
            materialization_event_sha256, "materialization_event_sha256"
        )
    return result


def _normalize_artifact_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("materialization requires a non-empty artifact inventory")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("materialization artifact inventory fields are invalid")
        relative = Path(_bounded_text(item["path"], "artifact inventory path", 500))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in {"", "."}:
            raise ValueError("materialization artifact inventory path is unsafe")
        size = item["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("materialization artifact size is invalid")
        result.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_text(item["sha256"], "artifact inventory sha256"),
                "size_bytes": size,
            }
        )
    result.sort(key=lambda item: item["path"])
    if len({item["path"] for item in result}) != len(result):
        raise ValueError("materialization artifact paths must be unique")
    return result


def exact_evaluation_materialized_payload(
    *,
    evaluation_run_id: str,
    members: Sequence[Mapping[str, Any]],
    report_core_sha256: str,
    envelope_sha256: str,
    artifact_inventory: Sequence[Mapping[str, Any]],
    authoritative_inventories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160),
        "members": _normalize_exact_members([dict(item) for item in members]),
        "report_core_sha256": _sha256_text(report_core_sha256, "report_core_sha256"),
        "envelope_sha256": _sha256_text(envelope_sha256, "envelope_sha256"),
        "artifact_inventory": _normalize_artifact_inventory(
            [dict(item) for item in artifact_inventory]
        ),
        "authoritative_inventories": _normalize_authoritative_inventories(
            [dict(item) for item in authoritative_inventories]
        ),
    }


def exact_evaluation_failed_payload(
    *, evaluation_run_id: str, reason_code: str, detail: str
) -> dict[str, Any]:
    return {
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160),
        "reason_code": _bounded_text(reason_code, "reason_code", 160),
        "detail": _bounded_text(detail, "failure detail", 2000),
        "promotion_eligible": False,
    }


def exact_promotion_decision_payload(
    *,
    evaluation_run_id: str,
    state: str,
    reason_codes: Sequence[str],
    hard_integrity_passed: bool,
    report_core_sha256: str,
    policy_sha256: str,
    decision_input_sha256: str,
    scorer_lock_sha256: str,
    environment_lock_sha256: str,
    manifest_sha256: str,
    members: Sequence[Mapping[str, Any]],
    failed_gate_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_state = _bounded_text(state, "promotion state", 40)
    if normalized_state not in {"promote", "review_required", "reject"}:
        raise ValueError(f"invalid exact promotion state: {normalized_state}")
    reasons = [_bounded_text(item, "promotion reason code", 160) for item in reason_codes]
    if len(reasons) != len(set(reasons)):
        raise ValueError("promotion reason codes must be unique")
    if normalized_state == "promote" and reasons:
        raise ValueError("promote cannot contain failed reason codes")
    if normalized_state != "promote" and not reasons:
        raise ValueError("non-promote decision requires reason codes")
    if normalized_state in {"promote", "review_required"} and hard_integrity_passed is not True:
        raise ValueError("non-reject decision requires passed hard integrity")
    if not isinstance(failed_gate_evidence, Mapping):
        raise ValueError("promotion failed-gate evidence must be an object")
    gate_evidence = {
        _bounded_text(name, "failed gate", 160): normalize_exact_values(value)
        for name, value in sorted(failed_gate_evidence.items(), key=lambda item: str(item[0]))
    }
    if set(gate_evidence) != set(reasons):
        raise ValueError("promotion failed-gate evidence must match reason codes exactly")
    return {
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160),
        "state": normalized_state,
        "reason_codes": reasons,
        "hard_integrity_passed": bool(hard_integrity_passed),
        "report_core_sha256": _sha256_text(report_core_sha256, "report_core_sha256"),
        "policy_sha256": _sha256_text(policy_sha256, "policy_sha256"),
        "decision_input_sha256": _sha256_text(
            decision_input_sha256, "decision_input_sha256"
        ),
        "scorer_lock_sha256": _sha256_text(scorer_lock_sha256, "scorer_lock_sha256"),
        "environment_lock_sha256": _sha256_text(
            environment_lock_sha256, "environment_lock_sha256"
        ),
        "manifest_sha256": _sha256_text(manifest_sha256, "manifest_sha256"),
        "members": _normalize_exact_members([dict(item) for item in members]),
        "failed_gate_evidence": gate_evidence,
        "authorized_for_submission": False,
    }


def exact_promotion_exception_payload(
    *,
    evaluation_run_id: str,
    decision_event_sha256: str,
    failed_gates: Mapping[str, Any],
    quantitative_tradeoff: str,
    approver: str,
    reason: str,
    downstream_authorization: str,
    report_core_sha256: str,
    policy_sha256: str,
    decision_input_sha256: str,
    scorer_lock_sha256: str,
    environment_lock_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(failed_gates, Mapping) or not failed_gates:
        raise ValueError("review exception requires failed-gate values")
    gates = {
        _bounded_text(name, "failed gate", 160): normalize_exact_values(value)
        for name, value in sorted(failed_gates.items(), key=lambda item: str(item[0]))
    }
    authorization = _bounded_text(
        downstream_authorization, "downstream_authorization", 500
    )
    if authorization not in REVIEW_EXCEPTION_AUTHORIZATIONS:
        raise ValueError(f"invalid downstream authorization: {authorization}")
    return {
        "evaluation_run_id": _bounded_text(evaluation_run_id, "evaluation_run_id", 160),
        "decision_event_sha256": _sha256_text(
            decision_event_sha256, "decision_event_sha256"
        ),
        "failed_gates": gates,
        "quantitative_tradeoff": _bounded_text(
            quantitative_tradeoff, "quantitative_tradeoff", 2000
        ),
        "approver": _bounded_text(approver, "approver", 240),
        "reason": _bounded_text(reason, "reason", 2000),
        "downstream_authorization": authorization,
        "report_core_sha256": _sha256_text(report_core_sha256, "report_core_sha256"),
        "policy_sha256": _sha256_text(policy_sha256, "policy_sha256"),
        "decision_input_sha256": _sha256_text(
            decision_input_sha256, "decision_input_sha256"
        ),
        "scorer_lock_sha256": _sha256_text(scorer_lock_sha256, "scorer_lock_sha256"),
        "environment_lock_sha256": _sha256_text(
            environment_lock_sha256, "environment_lock_sha256"
        ),
        "manifest_sha256": _sha256_text(manifest_sha256, "manifest_sha256"),
        "authorized_for_submission": False,
    }


def _validate_exact_transition(
    events: Sequence[ExperimentEvent], event: ExperimentEvent
) -> None:
    experiments = reconstruct_runs(events)
    evaluations = reconstruct_exact_evaluations(events)
    controls = reconstruct_cpu_acceptances(events)
    existing = evaluations.get(event.run_id)
    if event.payload.get("evaluation_run_id") != event.run_id:
        raise TransitionError("exact evaluation event/run identity mismatch")
    if event.event_type is EventType.EXACT_EVALUATION_REGISTERED:
        if existing is not None or event.run_id in experiments or event.run_id in controls:
            raise TransitionError(f"duplicate exact evaluation ID: {event.run_id}")
        try:
            normalized = exact_evaluation_registration_payload(
                evaluation_run_id=event.payload.get("evaluation_run_id"),
                scorer_lock_sha256=event.payload.get("scorer_lock_sha256"),
                environment_lock_sha256=event.payload.get("environment_lock_sha256"),
                manifest_sha256=event.payload.get("manifest_sha256"),
                evaluation_policy_sha256=event.payload.get("evaluation_policy_sha256"),
                evidence_kind=event.payload.get("evidence_kind"),
                members=event.payload.get("members", ()),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if set(event.payload) != set(normalized):
            raise TransitionError("exact evaluation registration has unknown fields")
        _validate_registered_exact_members(events, normalized["members"])
        return
    if existing is None:
        raise TransitionError(f"unknown exact evaluation ID: {event.run_id}")
    if event.event_type is EventType.EXACT_EVALUATION_STARTED:
        if existing.status is not ExactEvaluationStatus.REGISTERED:
            raise TransitionError(
                f"exact evaluation {event.run_id} cannot start from {existing.status}"
            )
        if event.payload != exact_evaluation_started_payload(evaluation_run_id=event.run_id):
            raise TransitionError("exact evaluation start payload is invalid")
        _validate_registered_exact_members(events, existing.registered["members"])
        return
    if existing.status is not ExactEvaluationStatus.RUNNING:
        raise TransitionError(
            f"exact evaluation {event.run_id} cannot finish from {existing.status}"
        )
    _validate_registered_exact_members(events, existing.registered["members"])
    if event.event_type is EventType.EXACT_EVALUATION_MATERIALIZED:
        if existing.materialized is not None:
            raise TransitionError("exact evaluation is already materialized")
        try:
            normalized_materialized = exact_evaluation_materialized_payload(
                evaluation_run_id=event.run_id,
                members=event.payload.get("members", ()),
                report_core_sha256=event.payload.get("report_core_sha256"),
                envelope_sha256=event.payload.get("envelope_sha256"),
                artifact_inventory=event.payload.get("artifact_inventory", ()),
                authoritative_inventories=event.payload.get(
                    "authoritative_inventories", ()
                ),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if event.payload != normalized_materialized:
            raise TransitionError("exact evaluation materialization payload is invalid")
        if canonical_json_bytes(normalized_materialized["members"]) != canonical_json_bytes(
            existing.registered["members"]
        ):
            raise TransitionError("exact evaluation materialization member table drifted")
        return
    if event.event_type is EventType.EXACT_EVALUATION_COMPLETED:
        try:
            normalized = exact_evaluation_completed_payload(
                evaluation_run_id=event.run_id,
                members=event.payload.get("members", ()),
                report_core_sha256=event.payload.get("report_core_sha256"),
                envelope_sha256=event.payload.get("envelope_sha256"),
                artifact_hashes=event.payload.get("artifact_hashes", {}),
                authoritative_inventories=event.payload.get("authoritative_inventories", ()),
                promotion_eligible=event.payload.get("promotion_eligible"),
                materialization_event_sha256=event.payload.get(
                    "materialization_event_sha256"
                ),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if event.payload != normalized:
            raise TransitionError("exact evaluation completion payload is invalid")
        if (
            existing.registered["evidence_kind"] == "official_data_control"
            and normalized["promotion_eligible"] is not False
        ):
            raise TransitionError("official-data control cannot be promotion eligible")
        if canonical_json_bytes(normalized["members"]) != canonical_json_bytes(
            existing.registered["members"]
        ):
            raise TransitionError("exact evaluation completion member table drifted")
        if "materialization_event_sha256" in normalized:
            if existing.materialized is None:
                raise TransitionError("exact evaluation completion requires materialization")
            materialized_event = next(
                item
                for item in existing.events
                if item.event_type is EventType.EXACT_EVALUATION_MATERIALIZED
            )
            if normalized["materialization_event_sha256"] != event_sha256(
                materialized_event
            ):
                raise TransitionError("exact evaluation materialization event hash mismatch")
            for name in (
                "members",
                "report_core_sha256",
                "envelope_sha256",
                "authoritative_inventories",
            ):
                if canonical_json_bytes(normalized[name]) != canonical_json_bytes(
                    existing.materialized[name]
                ):
                    raise TransitionError(
                        f"exact evaluation completion {name} was not materialized"
                    )
        registered_slots = {
            (item["role"], item["fold_id"], item["producer_run_id"])
            for item in existing.registered["members"]
        }
        inventory_slots = {
            (item["role"], item["fold_id"], item["producer_run_id"])
            for item in normalized["authoritative_inventories"]
        }
        if registered_slots != inventory_slots:
            raise TransitionError("exact evaluation authoritative inventory member drifted")
        return
    try:
        normalized_failure = exact_evaluation_failed_payload(
            evaluation_run_id=event.run_id,
            reason_code=event.payload.get("reason_code"),
            detail=event.payload.get("detail"),
        )
    except ValueError as exc:
        raise TransitionError(str(exc)) from exc
    if event.payload != normalized_failure:
        raise TransitionError("exact evaluation failure payload is invalid")


def _validate_cpu_transition(
    events: Sequence[ExperimentEvent], event: ExperimentEvent
) -> None:
    controls = reconstruct_cpu_acceptances(events)
    experiments = reconstruct_runs(events)
    evaluations = reconstruct_exact_evaluations(events)
    existing = controls.get(event.run_id)
    if event.event_type is EventType.CPU_ACCEPTANCE_REGISTERED:
        reserved_evaluation_ids = {
            str(state.registered["evaluation_run_id"]) for state in controls.values()
        }
        if (
            existing is not None
            or event.run_id in experiments
            or event.run_id in evaluations
            or event.run_id in reserved_evaluation_ids
        ):
            raise TransitionError(f"duplicate CPU acceptance ID: {event.run_id}")
        try:
            normalized = cpu_acceptance_registration_payload(
                run_id=event.payload.get("run_id"),
                purpose=event.payload.get("purpose"),
                request_nonce=event.payload.get("request_nonce"),
                acceptance_request_sha256=event.payload.get("acceptance_request_sha256"),
                evaluation_run_id=event.payload.get("evaluation_run_id"),
                kernel_slug=event.payload.get("kernel_slug"),
                runtime_dataset_slug=event.payload.get("runtime_dataset_slug"),
                runtime_bundle_name=event.payload.get("runtime_bundle_name"),
                runtime_bundle_sha256=event.payload.get("runtime_bundle_sha256"),
                runtime_bundle_inventory_sha256=event.payload.get(
                    "runtime_bundle_inventory_sha256"
                ),
                runtime_bundle_uncompressed_size_bytes=event.payload.get(
                    "runtime_bundle_uncompressed_size_bytes"
                ),
                runtime_bundle_file_count=event.payload.get(
                    "runtime_bundle_file_count"
                ),
                scorer_lock_sha256=event.payload.get("scorer_lock_sha256"),
                environment_lock_sha256=event.payload.get("environment_lock_sha256"),
                manifest_policy_sha256=event.payload.get("manifest_policy_sha256"),
                control_model_sha256=event.payload.get("control_model_sha256"),
                config_sha256=event.payload.get("config_sha256"),
                code_sha256=event.payload.get("code_sha256"),
                data_source_sha256=event.payload.get("data_source_sha256"),
                cpu_watchdog_minutes=event.payload.get("cpu_watchdog_minutes"),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if event.payload != normalized or event.run_id != normalized["run_id"]:
            raise TransitionError("CPU acceptance registration payload is invalid")
        if any(
            state.registered["request_nonce"] == normalized["request_nonce"]
            for state in controls.values()
        ):
            raise TransitionError("CPU acceptance request nonce has already been used")
        proposed_evaluation_id = normalized["evaluation_run_id"]
        if proposed_evaluation_id == normalized["run_id"]:
            raise TransitionError("CPU acceptance and proposed evaluation IDs must differ")
        if (
            proposed_evaluation_id in experiments
            or proposed_evaluation_id in evaluations
            or proposed_evaluation_id in controls
            or proposed_evaluation_id in reserved_evaluation_ids
        ):
            raise TransitionError("CPU acceptance proposed evaluation ID already exists")
        return
    if existing is None:
        raise TransitionError(f"unknown CPU acceptance ID: {event.run_id}")
    if event.event_type is EventType.CPU_ACCEPTANCE_FAILED:
        if existing.status in {CpuAcceptanceStatus.COMPLETED, CpuAcceptanceStatus.FAILED}:
            raise TransitionError(f"CPU acceptance {event.run_id} is already terminal")
        try:
            normalized_failure = cpu_acceptance_failed_payload(
                run_id=event.payload.get("run_id"),
                reason_code=event.payload.get("reason_code"),
                detail=event.payload.get("detail"),
                observed_artifact_hashes=event.payload.get("observed_artifact_hashes"),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if event.payload != normalized_failure or event.run_id != normalized_failure["run_id"]:
            raise TransitionError("CPU acceptance failure payload is invalid")
        return
    if event.event_type is EventType.CPU_ACCEPTANCE_STARTED:
        if existing.status is not CpuAcceptanceStatus.REGISTERED:
            raise TransitionError(f"CPU acceptance {event.run_id} cannot start")
        registered_event = next(
            item
            for item in existing.events
            if item.event_type is EventType.CPU_ACCEPTANCE_REGISTERED
        )
        try:
            normalized_start = cpu_acceptance_started_payload(
                run_id=event.payload.get("run_id"),
                registration_event_sha256=event.payload.get("registration_event_sha256"),
                kernel_ref=event.payload.get("kernel_ref"),
                runtime_dataset_ref=event.payload.get("runtime_dataset_ref"),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if event.payload != normalized_start or event.run_id != normalized_start["run_id"]:
            raise TransitionError("CPU acceptance start payload is invalid")
        if normalized_start["registration_event_sha256"] != event_sha256(registered_event):
            raise TransitionError("CPU acceptance start registration hash mismatch")
        if not normalized_start["kernel_ref"].startswith(existing.registered["kernel_slug"] + "/"):
            raise TransitionError("CPU acceptance started unexpected kernel")
        if not normalized_start["runtime_dataset_ref"].startswith(
            existing.registered["runtime_dataset_slug"] + "/"
        ):
            raise TransitionError("CPU acceptance started unexpected runtime dataset")
        return
    if event.event_type is EventType.CPU_ACCEPTANCE_INPUTS_BOUND:
        if existing.status is not CpuAcceptanceStatus.RUNNING:
            raise TransitionError(f"CPU acceptance {event.run_id} cannot bind inputs")
        try:
            normalized_binding = cpu_acceptance_inputs_bound_payload(
                run_id=event.payload.get("run_id"),
                manifest_sha256=event.payload.get("manifest_sha256"),
                folds=event.payload.get("folds", ()),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if event.payload != normalized_binding or event.run_id != normalized_binding["run_id"]:
            raise TransitionError("CPU acceptance input binding payload is invalid")
        return
    if existing.status is not CpuAcceptanceStatus.INPUTS_BOUND:
        raise TransitionError(f"CPU acceptance {event.run_id} cannot complete")
    try:
        normalized_completion = cpu_acceptance_completed_payload(
            run_id=event.payload.get("run_id"),
            actual_cpu_runtime_seconds=event.payload.get("actual_cpu_runtime_seconds"),
            peak_memory_mb=event.payload.get("peak_memory_mb"),
            remote_job_identity=event.payload.get("remote_job_identity"),
            graph_inventory_sha256=event.payload.get("graph_inventory_sha256"),
            artifact_hashes=event.payload.get("artifact_hashes", {}),
            output_inventory_sha256=event.payload.get("output_inventory_sha256"),
            pending_payload_sha256=event.payload.get("pending_payload_sha256"),
            pending_envelope_sha256=event.payload.get("pending_envelope_sha256"),
            reconciliation_sha256=event.payload.get("reconciliation_sha256"),
        )
    except ValueError as exc:
        raise TransitionError(str(exc)) from exc
    if event.payload != normalized_completion or event.run_id != normalized_completion["run_id"]:
        raise TransitionError("CPU acceptance completion payload is invalid")


def _validate_exact_decision_transition(
    events: Sequence[ExperimentEvent], event: ExperimentEvent
) -> None:
    evaluations = reconstruct_exact_evaluations(events)
    state = evaluations.get(event.run_id)
    if state is None:
        raise TransitionError(f"decision for unknown exact evaluation {event.run_id}")
    if state.status is not ExactEvaluationStatus.COMPLETED or state.terminal is None:
        raise TransitionError("exact promotion requires a completed aggregate evaluation")
    if state.registered.get("evidence_kind") != "model_candidate":
        raise TransitionError("non-candidate exact evidence cannot receive a promotion decision")
    _validate_registered_exact_members(events, state.registered["members"])
    if event.event_type is EventType.EXACT_PROMOTION_DECISION:
        if state.decision is not None:
            raise TransitionError("exact evaluation already has a promotion decision")
        try:
            normalized = exact_promotion_decision_payload(
                evaluation_run_id=event.payload.get("evaluation_run_id"),
                state=event.payload.get("state"),
                reason_codes=event.payload.get("reason_codes", ()),
                hard_integrity_passed=event.payload.get("hard_integrity_passed"),
                report_core_sha256=event.payload.get("report_core_sha256"),
                policy_sha256=event.payload.get("policy_sha256"),
                decision_input_sha256=event.payload.get("decision_input_sha256"),
                scorer_lock_sha256=event.payload.get("scorer_lock_sha256"),
                environment_lock_sha256=event.payload.get("environment_lock_sha256"),
                manifest_sha256=event.payload.get("manifest_sha256"),
                members=event.payload.get("members", ()),
                failed_gate_evidence=event.payload.get("failed_gate_evidence", {}),
            )
        except ValueError as exc:
            raise TransitionError(str(exc)) from exc
        if event.payload != normalized or event.run_id != normalized["evaluation_run_id"]:
            raise TransitionError("exact promotion decision payload is invalid")
        for name in (
            "report_core_sha256",
            "scorer_lock_sha256",
            "environment_lock_sha256",
            "manifest_sha256",
        ):
            expected = (
                state.terminal[name]
                if name == "report_core_sha256"
                else state.registered[name]
            )
            if normalized[name] != expected:
                raise TransitionError(f"exact promotion {name} mismatch")
        if canonical_json_bytes(normalized["members"]) != canonical_json_bytes(
            state.registered["members"]
        ):
            raise TransitionError("exact promotion member table mismatch")
        return
    if state.decision is None or state.decision.get("state") != "review_required":
        raise TransitionError("exception requires an immutable review_required decision")
    if state.exceptions:
        raise TransitionError("exact review exception is already recorded")
    decision_event = next(
        item
        for item in state.events
        if item.event_type is EventType.EXACT_PROMOTION_DECISION
    )
    try:
        normalized_exception = exact_promotion_exception_payload(
            evaluation_run_id=event.payload.get("evaluation_run_id"),
            decision_event_sha256=event.payload.get("decision_event_sha256"),
            failed_gates=event.payload.get("failed_gates", {}),
            quantitative_tradeoff=event.payload.get("quantitative_tradeoff"),
            approver=event.payload.get("approver"),
            reason=event.payload.get("reason"),
            downstream_authorization=event.payload.get("downstream_authorization"),
            report_core_sha256=event.payload.get("report_core_sha256"),
            policy_sha256=event.payload.get("policy_sha256"),
            decision_input_sha256=event.payload.get("decision_input_sha256"),
            scorer_lock_sha256=event.payload.get("scorer_lock_sha256"),
            environment_lock_sha256=event.payload.get("environment_lock_sha256"),
            manifest_sha256=event.payload.get("manifest_sha256"),
        )
    except ValueError as exc:
        raise TransitionError(str(exc)) from exc
    if event.payload != normalized_exception or event.run_id != normalized_exception["evaluation_run_id"]:
        raise TransitionError("exact promotion exception payload is invalid")
    if normalized_exception["decision_event_sha256"] != event_sha256(decision_event):
        raise TransitionError("exact promotion exception decision hash mismatch")
    expected_gates = state.decision.get("failed_gate_evidence")
    if (
        not isinstance(expected_gates, Mapping)
        or set(normalized_exception["failed_gates"])
        != set(state.decision.get("reason_codes", ()))
        or canonical_json_bytes(normalized_exception["failed_gates"])
        != canonical_json_bytes(expected_gates)
    ):
        raise TransitionError("exact promotion exception failed gates do not match decision")
    for name in (
        "report_core_sha256",
        "policy_sha256",
        "decision_input_sha256",
        "scorer_lock_sha256",
        "environment_lock_sha256",
        "manifest_sha256",
    ):
        if normalized_exception[name] != state.decision[name]:
            raise TransitionError(f"exact promotion exception {name} mismatch")


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
