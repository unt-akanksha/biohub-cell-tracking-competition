from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import canonical_json_bytes, sha256_bytes
from .ledger import (
    CpuAcceptanceStatus,
    EventType,
    Ledger,
    RunStatus,
    event_sha256,
    reconstruct_runs,
    reconstruct_cpu_acceptances,
    validate_producer_registration_evidence,
    validate_producer_terminal_evidence,
)


class EvidenceError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> None:
    raise EvidenceError(reason_code, detail)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("PREDICTION_SET_SCHEMA_INVALID", f"{name} must be an object")
    return value


def _sha256(value: Any, name: str) -> str:
    digest = str(value).casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        _fail("PREDICTION_SET_SCHEMA_INVALID", f"{name} must be a SHA-256 digest")
    return digest


def _text(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result or len(result) > 240:
        _fail("PREDICTION_SET_SCHEMA_INVALID", f"{name} is invalid")
    return result


@dataclass(frozen=True)
class GraphArtifact:
    sample_id: str
    path: str
    sha256: str
    producer_run_id: str
    fold_id: str

    @classmethod
    def from_dict(cls, value: Any) -> "GraphArtifact":
        item = _mapping(value, "graph artifact")
        return cls(
            sample_id=_text(item.get("sample_id"), "graphs.sample_id"),
            path=_text(item.get("path"), "graphs.path"),
            sha256=_sha256(item.get("sha256"), "graphs.sha256"),
            producer_run_id=_text(item.get("producer_run_id"), "graphs.producer_run_id"),
            fold_id=_text(item.get("fold_id"), "graphs.fold_id"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "sha256": self.sha256,
            "producer_run_id": self.producer_run_id,
            "fold_id": self.fold_id,
        }


@dataclass(frozen=True)
class PredictionSetClaim:
    producer_run_id: str
    fold_id: str
    manifest_sha256: str
    train_membership_sha256: str
    calibration_membership_sha256: str
    evaluation_membership_sha256: str
    model_sha256: str
    config_sha256: str
    code_sha256: str
    data_sha256: str
    graphs: tuple[GraphArtifact, ...]
    graph_inventory_sha256: str
    artifact_hashes: tuple[tuple[str, str], ...]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: Any) -> "PredictionSetClaim":
        data = _mapping(value, "prediction set")
        if data.get("schema_version") != 1:
            _fail("PREDICTION_SET_SCHEMA_INVALID", "schema_version must be 1")
        raw_graphs = data.get("graphs")
        if not isinstance(raw_graphs, list) or not raw_graphs:
            _fail("PREDICTION_SET_SCHEMA_INVALID", "graphs must be a non-empty array")
        graphs = tuple(GraphArtifact.from_dict(item) for item in raw_graphs)
        if graphs != tuple(sorted(graphs, key=lambda item: item.sample_id)):
            _fail("PREDICTION_SET_SCHEMA_INVALID", "graphs must be sorted by sample_id")
        if len({item.sample_id for item in graphs}) != len(graphs):
            _fail("DUPLICATE_GRAPH_IDENTITY", "graph sample IDs must be unique")
        raw_artifacts = _mapping(data.get("artifact_hashes"), "artifact_hashes")
        if not raw_artifacts:
            _fail("PREDICTION_SET_SCHEMA_INVALID", "artifact_hashes cannot be empty")
        artifact_hashes = tuple(
            (str(name), _sha256(raw_artifacts[name], f"artifact_hashes.{name}"))
            for name in sorted(raw_artifacts, key=str)
        )
        claim = cls(
            producer_run_id=_text(data.get("producer_run_id"), "producer_run_id"),
            fold_id=_text(data.get("fold_id"), "fold_id"),
            manifest_sha256=_sha256(data.get("manifest_sha256"), "manifest_sha256"),
            train_membership_sha256=_sha256(
                data.get("train_membership_sha256"), "train_membership_sha256"
            ),
            calibration_membership_sha256=_sha256(
                data.get("calibration_membership_sha256"), "calibration_membership_sha256"
            ),
            evaluation_membership_sha256=_sha256(
                data.get("evaluation_membership_sha256"), "evaluation_membership_sha256"
            ),
            model_sha256=_sha256(data.get("model_sha256"), "model_sha256"),
            config_sha256=_sha256(data.get("config_sha256"), "config_sha256"),
            code_sha256=_sha256(data.get("code_sha256"), "code_sha256"),
            data_sha256=_sha256(data.get("data_sha256"), "data_sha256"),
            graphs=graphs,
            graph_inventory_sha256=_sha256(
                data.get("graph_inventory_sha256"), "graph_inventory_sha256"
            ),
            artifact_hashes=artifact_hashes,
        )
        if claim.computed_inventory_sha256() != claim.graph_inventory_sha256:
            _fail("GRAPH_INVENTORY_HASH_MISMATCH", "declared graph inventory is not canonical")
        return claim

    def computed_inventory_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes([item.to_dict() for item in self.graphs]))

    def registration_evidence(self) -> dict[str, str]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "fold_id": self.fold_id,
            "train_membership_sha256": self.train_membership_sha256,
            "calibration_membership_sha256": self.calibration_membership_sha256,
            "evaluation_membership_sha256": self.evaluation_membership_sha256,
            "model_sha256": self.model_sha256,
            "config_sha256": self.config_sha256,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
        }

    def terminal_evidence(self) -> dict[str, Any]:
        return {
            "evidence_eligible": True,
            "graph_inventory_sha256": self.graph_inventory_sha256,
            "artifact_hashes": dict(self.artifact_hashes),
        }


@dataclass(frozen=True)
class ResolvedProducer:
    run_id: str
    registered: Mapping[str, Any]
    terminal: Mapping[str, Any]
    registration_event_sha256: str
    terminal_event_sha256: str
    input_binding_event_sha256: str | None = None


def load_prediction_set(path: str | Path) -> PredictionSetClaim:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceError("PREDICTION_SET_MISSING", str(path)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError("PREDICTION_SET_UNREADABLE", str(path)) from exc
    return PredictionSetClaim.from_dict(value)


def _equal(left: Any, right: Any) -> bool:
    return secrets.compare_digest(canonical_json_bytes(left), canonical_json_bytes(right))


def resolve_producer(
    claim: PredictionSetClaim,
    ledger: Ledger | Sequence[Any],
) -> ResolvedProducer:
    """Resolve an untrusted sidecar to immutable registration and terminal events."""

    events = ledger.read_events() if isinstance(ledger, Ledger) else list(ledger)
    runs = reconstruct_runs(events)
    state = runs.get(claim.producer_run_id)
    if state is None:
        controls = reconstruct_cpu_acceptances(events)
        control = controls.get(claim.producer_run_id)
        if control is None:
            _fail("UNKNOWN_PRODUCER", claim.producer_run_id)
        if control.status is not CpuAcceptanceStatus.COMPLETED or control.terminal is None:
            _fail("NONTERMINAL_PRODUCER", claim.producer_run_id)
        if control.inputs_bound is None:
            _fail("PRODUCER_NOT_EVIDENCE_ELIGIBLE", claim.producer_run_id)
        fold = next(
            (
                item
                for item in control.inputs_bound["folds"]
                if item["fold_id"] == claim.fold_id
            ),
            None,
        )
        if fold is None:
            _fail("REGISTERED_LINEAGE_MISMATCH", claim.producer_run_id)
        registered = {
            "manifest_sha256": control.inputs_bound["manifest_sha256"],
            "fold_id": claim.fold_id,
            "train_membership_sha256": fold["train_membership_sha256"],
            "calibration_membership_sha256": fold["calibration_membership_sha256"],
            "evaluation_membership_sha256": fold["evaluation_membership_sha256"],
            "model_sha256": control.registered["control_model_sha256"],
            "config_sha256": control.registered["config_sha256"],
            "code_sha256": control.registered["code_sha256"],
            "data_sha256": control.registered["data_source_sha256"],
        }
        terminal = {
            "evidence_eligible": True,
            "graph_inventory_sha256": control.terminal["graph_inventory_sha256"],
            "artifact_hashes": control.terminal["artifact_hashes"],
        }
        if not _equal(registered, claim.registration_evidence()):
            _fail("REGISTERED_LINEAGE_MISMATCH", claim.producer_run_id)
        if not _equal(terminal, claim.terminal_evidence()):
            _fail("TERMINAL_ARTIFACT_MISMATCH", claim.producer_run_id)
        registered_event = next(
            item
            for item in control.events
            if item.event_type is EventType.CPU_ACCEPTANCE_REGISTERED
        )
        binding_event = next(
            item
            for item in control.events
            if item.event_type is EventType.CPU_ACCEPTANCE_INPUTS_BOUND
        )
        terminal_event = next(
            item
            for item in control.events
            if item.event_type is EventType.CPU_ACCEPTANCE_COMPLETED
        )
        return ResolvedProducer(
            claim.producer_run_id,
            registered,
            terminal,
            event_sha256(registered_event),
            event_sha256(terminal_event),
            event_sha256(binding_event),
        )
    if state.status is not RunStatus.COMPLETED or state.terminal is None:
        _fail("NONTERMINAL_PRODUCER", claim.producer_run_id)
    try:
        registered = validate_producer_registration_evidence(state.registered, required=True)
    except ValueError as exc:
        raise EvidenceError("PRODUCER_NOT_EVIDENCE_ELIGIBLE", str(exc)) from exc
    try:
        terminal = validate_producer_terminal_evidence(state.terminal, required=True)
    except ValueError as exc:
        raise EvidenceError("PRODUCER_NOT_EVIDENCE_ELIGIBLE", str(exc)) from exc
    if not _equal(registered, claim.registration_evidence()):
        _fail("REGISTERED_LINEAGE_MISMATCH", claim.producer_run_id)
    if not _equal(terminal, claim.terminal_evidence()):
        _fail("TERMINAL_ARTIFACT_MISMATCH", claim.producer_run_id)
    registered_event = next(
        item for item in state.events if item.event_type is EventType.REGISTERED
    )
    terminal_event = next(
        item for item in state.events if item.event_type is EventType.COMPLETED
    )
    return ResolvedProducer(
        claim.producer_run_id,
        state.registered,
        state.terminal,
        event_sha256(registered_event),
        event_sha256(terminal_event),
    )
