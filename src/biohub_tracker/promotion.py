from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import ExactEvaluationError, ExactReport, validate_exact_core
from .io import canonical_json_bytes, sha256_bytes
from .ledger import (
    EventType,
    ExactEvaluationStatus,
    ExperimentEvent,
    Ledger,
    TransitionError,
    event_sha256,
    exact_promotion_decision_payload,
    exact_promotion_exception_payload,
    reconstruct_exact_evaluations,
    resolved_exact_member,
)


POLICY_SCHEMA = "biohub.promotion-policy.v1"
INPUT_SCHEMA = "biohub.promotion-input.v1"
_POLICY_KEYS = {
    "schema_version",
    "decision_input_schema",
    "report_schema",
    "required_evidence_kind",
    "hard_gate_order",
    "soft_gate_order",
    "required_integrity",
    "thresholds",
    "decision_input_allowlist",
    "policy_sha256",
}
_THRESHOLD_KEYS = {
    "pooled_final_score_delta_min_exclusive",
    "bilateral_score_delta_min",
    "bootstrap_lower_score_delta_min",
    "bootstrap_probability_min",
    "node_recall_micro_delta_min",
    "division_jaccard_delta_min",
    "division_tp_delta_min",
    "worst_movie_delta_min",
}
_EXPECTED_INTEGRITY = {
    "producer_ledger_resolution": "passed",
    "aggregate_member_match": "passed",
    "complete_coverage": "passed",
    "native_graph_integrity": "passed",
    "submission_roundtrip": "passed",
    "official_count_parity": "passed",
    "metric_exploit_audit": "passed",
    "authoritative_prediction_space": "integer-csv-rebuilt-geff",
    "native_prediction_space": "diagnostic_only",
}
_HARD_GATE_ORDER = (
    "NON_CANDIDATE_EVIDENCE_KIND",
    "EXACT_EVALUATION_NOT_COMPLETED",
    "EXACT_REPORT_HASH_MISMATCH",
    "EXACT_EVALUATION_REGISTRATION_MISMATCH",
    "UNKNOWN_PRODUCER",
    "NONTERMINAL_PRODUCER",
    "PRODUCER_NOT_EVIDENCE_ELIGIBLE",
    "REGISTERED_LINEAGE_MISMATCH",
    "TERMINAL_ARTIFACT_MISMATCH",
    "SCORER_SOURCE_MISMATCH",
    "MANIFEST_MISMATCH",
    "METRIC_EXPLOIT_SIGNATURE",
    "INCOMPLETE_MOVIE_COVERAGE",
    "GRAPH_INTEGRITY_FAILURE",
    "ROUNDTRIP_INVALID",
    "OFFICIAL_COUNT_PARITY_FAILURE",
)
_SOFT_GATE_ORDER = (
    "POOLED_SCORE_NOT_POSITIVE",
    "BILATERAL_EMBRYO_REGRESSION",
    "BOOTSTRAP_LOWER_BOUND_UNSTABLE",
    "BOOTSTRAP_PROBABILITY_LOW",
    "NODE_RECALL_REGRESSION",
    "DIVISION_JACCARD_REGRESSION",
    "DIVISION_TP_REGRESSION",
    "WORST_MOVIE_COLLAPSE",
)
_ALLOWLIST = (
    "evaluation_run_id",
    "evidence_kind",
    "scorer_lock_sha256",
    "environment_lock_sha256",
    "manifest_sha256",
    "evaluation_policy_sha256",
    "report_core_sha256",
    "aggregate_evaluation",
    "members",
    "authoritative_inventories",
    "coverage",
    "integrity_checks",
    "comparison.pooled",
    "comparison.by_embryo",
    "comparison.by_fold",
    "comparison.worst_paired_movie",
    "comparison.bootstrap",
)


class PromotionError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> None:
    raise PromotionError(reason_code, detail)


def _sha(value: Any, name: str) -> str:
    digest = str(value).casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", name)
    return digest


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        _fail("PROMOTION_DECISION_INPUT_INVALID", name)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PromotionError("PROMOTION_DECISION_INPUT_INVALID", name) from exc
    if not result.is_finite():
        _fail("PROMOTION_DECISION_INPUT_INVALID", name)
    return result


@dataclass(frozen=True)
class PromotionPolicy:
    raw: Mapping[str, Any]
    policy_sha256: str
    thresholds: Mapping[str, Any]
    hard_gate_order: tuple[str, ...]
    soft_gate_order: tuple[str, ...]


@dataclass(frozen=True)
class PromotionDecision:
    state: str
    reason_codes: tuple[str, ...]
    hard_failures: tuple[str, ...]
    soft_failures: tuple[str, ...]
    soft_gates_evaluated: bool
    evaluation_run_id: str
    report_core_sha256: str
    policy_sha256: str
    decision_input_sha256: str
    decision_inputs: Mapping[str, Any]
    schema_version: str = "biohub.promotion-decision.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "hard_failures": list(self.hard_failures),
            "soft_failures": list(self.soft_failures),
            "soft_gates_evaluated": self.soft_gates_evaluated,
            "evaluation_run_id": self.evaluation_run_id,
            "report_core_sha256": self.report_core_sha256,
            "policy_sha256": self.policy_sha256,
            "decision_input_sha256": self.decision_input_sha256,
            "decision_inputs": dict(self.decision_inputs),
        }


def load_promotion_policy(path: str | Path) -> PromotionPolicy:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError("PROMOTION_POLICY_UNREADABLE", str(path)) from exc
    if not isinstance(value, dict) or set(value) != _POLICY_KEYS:
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", "unknown or missing root field")
    if (
        value["schema_version"] != POLICY_SCHEMA
        or value["decision_input_schema"] != INPUT_SCHEMA
        or value["report_schema"] != "biohub.exact-report.v2"
        or value["required_evidence_kind"] != "model_candidate"
    ):
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", "schema or evidence kind")
    if tuple(value["hard_gate_order"]) != _HARD_GATE_ORDER:
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", "hard gate order")
    if tuple(value["soft_gate_order"]) != _SOFT_GATE_ORDER:
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", "soft gate order")
    if tuple(value["decision_input_allowlist"]) != _ALLOWLIST:
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", "decision input allowlist")
    if value["required_integrity"] != _EXPECTED_INTEGRITY:
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", "required integrity boundary")
    thresholds = value["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != _THRESHOLD_KEYS:
        _fail("PROMOTION_POLICY_SCHEMA_INVALID", "threshold schema")
    for name in sorted(_THRESHOLD_KEYS):
        _decimal(thresholds[name], f"thresholds.{name}")
    semantic = dict(value)
    declared = _sha(semantic.pop("policy_sha256"), "policy_sha256")
    computed = sha256_bytes(canonical_json_bytes(semantic))
    if not secrets.compare_digest(declared, computed):
        _fail("PROMOTION_POLICY_HASH_MISMATCH", "intrinsic policy hash")
    return PromotionPolicy(value, declared, thresholds, _HARD_GATE_ORDER, _SOFT_GATE_ORDER)


def _mapped_resolution_reason(exc: Exception) -> str:
    text = str(exc).casefold()
    if "unknown exact-evaluation producer" in text:
        return "UNKNOWN_PRODUCER"
    if "nonterminal exact-evaluation producer" in text:
        return "NONTERMINAL_PRODUCER"
    if "ineligible exact-evaluation producer" in text:
        return "PRODUCER_NOT_EVIDENCE_ELIGIBLE"
    return "EXACT_EVALUATION_REGISTRATION_MISMATCH"


def _ordered(reasons: Sequence[str], order: Sequence[str]) -> tuple[str, ...]:
    present = set(reasons)
    return tuple(reason for reason in order if reason in present)


def _hard_failures(
    *,
    report: ExactReport,
    ledger: Ledger,
    evaluation_run_id: str,
    policy: PromotionPolicy,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    core = report.core
    events = ledger.read_events()
    evaluations = reconstruct_exact_evaluations(events)
    state = evaluations.get(evaluation_run_id)
    reasons: list[str] = []
    if core["evidence_kind"] != policy.raw["required_evidence_kind"]:
        reasons.append("NON_CANDIDATE_EVIDENCE_KIND")
    if state is None or state.status is not ExactEvaluationStatus.COMPLETED or state.terminal is None:
        reasons.append("EXACT_EVALUATION_NOT_COMPLETED")
    else:
        identities = (
            ("scorer_lock_sha256", "SCORER_SOURCE_MISMATCH"),
            ("environment_lock_sha256", "SCORER_SOURCE_MISMATCH"),
            ("manifest_sha256", "MANIFEST_MISMATCH"),
            ("evaluation_policy_sha256", "SCORER_SOURCE_MISMATCH"),
        )
        for name, reason in identities:
            if state.registered.get(name) != core.get(name):
                reasons.append(reason)
        if canonical_json_bytes(state.registered.get("members")) != canonical_json_bytes(
            core.get("members")
        ):
            reasons.append("EXACT_EVALUATION_REGISTRATION_MISMATCH")
        if state.terminal.get("report_core_sha256") != report.core_sha256:
            reasons.append("EXACT_REPORT_HASH_MISMATCH")
        if canonical_json_bytes(state.terminal.get("authoritative_inventories")) != canonical_json_bytes(
            core.get("authoritative_inventories")
        ):
            reasons.append("ROUNDTRIP_INVALID")

    for member in core["members"]:
        try:
            resolved = resolved_exact_member(
                events,
                role=str(member["role"]),
                fold_id=str(member["fold_id"]),
                producer_run_id=str(member["producer_run_id"]),
            )
        except TransitionError as exc:
            reasons.append(_mapped_resolution_reason(exc))
            continue
        registration_names = (
            "manifest_sha256",
            "fold_id",
            "train_membership_sha256",
            "calibration_membership_sha256",
            "evaluation_membership_sha256",
            "model_sha256",
            "config_sha256",
            "code_sha256",
            "data_sha256",
            "producer_registration_event_sha256",
        )
        terminal_names = (
            "graph_inventory_sha256",
            "artifact_hashes",
            "producer_terminal_event_sha256",
        )
        if any(
            canonical_json_bytes(resolved.get(name)) != canonical_json_bytes(member.get(name))
            for name in registration_names
        ):
            reasons.append("REGISTERED_LINEAGE_MISMATCH")
        elif any(
            canonical_json_bytes(resolved.get(name)) != canonical_json_bytes(member.get(name))
            for name in terminal_names
        ):
            reasons.append("TERMINAL_ARTIFACT_MISMATCH")

    integrity = core["integrity_checks"]
    if integrity.get("metric_exploit_audit") != "passed":
        reasons.append("METRIC_EXPLOIT_SIGNATURE")
    if integrity.get("complete_coverage") != "passed" or core["coverage"].get("complete") is not True:
        reasons.append("INCOMPLETE_MOVIE_COVERAGE")
    if integrity.get("native_graph_integrity") != "passed":
        reasons.append("GRAPH_INTEGRITY_FAILURE")
    if (
        integrity.get("submission_roundtrip") != "passed"
        or integrity.get("authoritative_prediction_space") != "integer-csv-rebuilt-geff"
        or integrity.get("native_prediction_space") != "diagnostic_only"
    ):
        reasons.append("ROUNDTRIP_INVALID")
    if integrity.get("official_count_parity") != "passed":
        reasons.append("OFFICIAL_COUNT_PARITY_FAILURE")
    if integrity.get("producer_ledger_resolution") != "passed":
        reasons.append("REGISTERED_LINEAGE_MISMATCH")
    if integrity.get("aggregate_member_match") != "passed":
        reasons.append("EXACT_EVALUATION_REGISTRATION_MISMATCH")

    registration_event = next(
        (
            event
            for event in events
            if event.run_id == evaluation_run_id
            and event.event_type is EventType.EXACT_EVALUATION_REGISTERED
        ),
        None,
    )
    terminal_event = next(
        (
            event
            for event in events
            if event.run_id == evaluation_run_id
            and event.event_type is EventType.EXACT_EVALUATION_COMPLETED
        ),
        None,
    )
    aggregate = {
        "evaluation_run_id": evaluation_run_id,
        "status": state.status.value if state is not None else "unknown",
        "registration_event_sha256": event_sha256(registration_event)
        if registration_event is not None
        else None,
        "terminal_event_sha256": event_sha256(terminal_event)
        if terminal_event is not None
        else None,
    }
    return _ordered(reasons, policy.hard_gate_order), aggregate


def _delta(group: Mapping[str, Any], metric: str) -> Decimal:
    try:
        value = group["metrics"][metric]
    except (KeyError, TypeError) as exc:
        raise PromotionError("PROMOTION_DECISION_INPUT_INVALID", metric) from exc
    if not isinstance(value, Mapping) or value.get("status") != "applicable":
        _fail("PROMOTION_DECISION_INPUT_INVALID", f"{metric} is not comparable")
    return _decimal(value.get("delta"), metric)


def _soft_failures(core: Mapping[str, Any], policy: PromotionPolicy) -> tuple[str, ...]:
    comparison = core["comparison"]
    pooled = comparison["pooled"]
    thresholds = policy.thresholds
    reasons: list[str] = []
    if _delta(pooled, "score") <= _decimal(
        thresholds["pooled_final_score_delta_min_exclusive"], "pooled threshold"
    ):
        reasons.append("POOLED_SCORE_NOT_POSITIVE")
    bilateral_floor = _decimal(thresholds["bilateral_score_delta_min"], "bilateral threshold")
    by_embryo = comparison.get("by_embryo")
    if not isinstance(by_embryo, Mapping) or len(by_embryo) != 2:
        _fail("PROMOTION_DECISION_INPUT_INVALID", "two embryo groups are required")
    if any(_delta(by_embryo[name], "score") < bilateral_floor for name in sorted(by_embryo)):
        reasons.append("BILATERAL_EMBRYO_REGRESSION")
    try:
        bootstrap_score = comparison["bootstrap"]["metrics"]["score"]
        lower = bootstrap_score["percentiles"]["2.5"]
        probability = bootstrap_score["probability_candidate_gt_baseline"]
    except (KeyError, TypeError) as exc:
        raise PromotionError("PROMOTION_DECISION_INPUT_INVALID", "bootstrap score") from exc
    if _decimal(lower, "bootstrap lower") < _decimal(
        thresholds["bootstrap_lower_score_delta_min"], "bootstrap lower threshold"
    ):
        reasons.append("BOOTSTRAP_LOWER_BOUND_UNSTABLE")
    if _decimal(probability, "bootstrap probability") < _decimal(
        thresholds["bootstrap_probability_min"], "bootstrap probability threshold"
    ):
        reasons.append("BOOTSTRAP_PROBABILITY_LOW")
    if _delta(pooled, "node_recall_micro") < _decimal(
        thresholds["node_recall_micro_delta_min"], "node threshold"
    ):
        reasons.append("NODE_RECALL_REGRESSION")
    division = pooled["metrics"].get("division_jaccard")
    if not isinstance(division, Mapping):
        _fail("PROMOTION_DECISION_INPUT_INVALID", "division_jaccard")
    if division.get("status") == "applicable":
        if _decimal(division.get("delta"), "division_jaccard") < _decimal(
            thresholds["division_jaccard_delta_min"], "division threshold"
        ):
            reasons.append("DIVISION_JACCARD_REGRESSION")
    elif division.get("status") != "not_applicable":
        reasons.append("DIVISION_JACCARD_REGRESSION")
    try:
        division_tp = pooled["counts"]["division_tp"]["delta"]
    except (KeyError, TypeError) as exc:
        raise PromotionError("PROMOTION_DECISION_INPUT_INVALID", "division_tp") from exc
    if _decimal(division_tp, "division_tp") < _decimal(
        thresholds["division_tp_delta_min"], "division TP threshold"
    ):
        reasons.append("DIVISION_TP_REGRESSION")
    try:
        worst = comparison["worst_paired_movie"]["metrics"]["score"]["delta"]
    except (KeyError, TypeError) as exc:
        raise PromotionError("PROMOTION_DECISION_INPUT_INVALID", "worst movie") from exc
    if _decimal(worst, "worst movie") < _decimal(
        thresholds["worst_movie_delta_min"], "worst threshold"
    ):
        reasons.append("WORST_MOVIE_COLLAPSE")
    return _ordered(reasons, policy.soft_gate_order)


def _decision_inputs(
    core: Mapping[str, Any],
    *,
    report_core_sha256: str,
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = core["comparison"]
    return {
        "schema_version": INPUT_SCHEMA,
        "evaluation_run_id": core["evaluation_run_id"],
        "evidence_kind": core["evidence_kind"],
        "scorer_lock_sha256": core["scorer_lock_sha256"],
        "environment_lock_sha256": core["environment_lock_sha256"],
        "manifest_sha256": core["manifest_sha256"],
        "evaluation_policy_sha256": core["evaluation_policy_sha256"],
        "report_core_sha256": report_core_sha256,
        "aggregate_evaluation": dict(aggregate),
        "members": list(core["members"]),
        "authoritative_inventories": list(core["authoritative_inventories"]),
        "coverage": dict(core["coverage"]),
        "integrity_checks": dict(core["integrity_checks"]),
        "comparison": {
            "pooled": comparison["pooled"],
            "by_embryo": comparison["by_embryo"],
            "by_fold": comparison["by_fold"],
            "worst_paired_movie": comparison["worst_paired_movie"],
            "bootstrap": comparison["bootstrap"],
        },
    }


def _failed_gate_evidence(
    decision: PromotionDecision, policy: PromotionPolicy
) -> dict[str, Any]:
    comparison = decision.decision_inputs["comparison"]
    pooled = comparison["pooled"]
    thresholds = policy.thresholds
    evidence: dict[str, Any] = {}
    for reason in decision.reason_codes:
        if reason == "POOLED_SCORE_NOT_POSITIVE":
            observed = pooled["metrics"]["score"]["delta"]
            threshold = thresholds["pooled_final_score_delta_min_exclusive"]
        elif reason == "BILATERAL_EMBRYO_REGRESSION":
            observed = {
                name: group["metrics"]["score"]["delta"]
                for name, group in sorted(comparison["by_embryo"].items())
            }
            threshold = thresholds["bilateral_score_delta_min"]
        elif reason == "BOOTSTRAP_LOWER_BOUND_UNSTABLE":
            observed = comparison["bootstrap"]["metrics"]["score"]["percentiles"]["2.5"]
            threshold = thresholds["bootstrap_lower_score_delta_min"]
        elif reason == "BOOTSTRAP_PROBABILITY_LOW":
            observed = comparison["bootstrap"]["metrics"]["score"][
                "probability_candidate_gt_baseline"
            ]
            threshold = thresholds["bootstrap_probability_min"]
        elif reason == "NODE_RECALL_REGRESSION":
            observed = pooled["metrics"]["node_recall_micro"]["delta"]
            threshold = thresholds["node_recall_micro_delta_min"]
        elif reason == "DIVISION_JACCARD_REGRESSION":
            observed = pooled["metrics"]["division_jaccard"]
            threshold = thresholds["division_jaccard_delta_min"]
        elif reason == "DIVISION_TP_REGRESSION":
            observed = pooled["counts"]["division_tp"]["delta"]
            threshold = thresholds["division_tp_delta_min"]
        elif reason == "WORST_MOVIE_COLLAPSE":
            observed = comparison["worst_paired_movie"]["metrics"]["score"]["delta"]
            threshold = thresholds["worst_movie_delta_min"]
        else:
            observed = "failed"
            threshold = "hard-integrity-pass-required"
        evidence[reason] = {
            "observed": observed,
            "threshold": threshold,
            "decision_input_sha256": decision.decision_input_sha256,
        }
    return evidence


def evaluate_promotion(
    report: ExactReport,
    *,
    policy_path: str | Path,
    ledger_path: str | Path,
    workspace_root: str | Path,
    evaluation_run_id: str,
) -> PromotionDecision:
    policy = load_promotion_policy(policy_path)
    try:
        core = validate_exact_core(dict(report.core))
    except ExactEvaluationError as exc:
        raise PromotionError("PROMOTION_REPORT_SCHEMA_INVALID", str(exc)) from exc
    if core["schema_version"] != policy.raw["report_schema"]:
        _fail("PROMOTION_REPORT_SCHEMA_INVALID", "legacy report is audit-only")
    computed_core_sha = sha256_bytes(canonical_json_bytes(core))
    if not secrets.compare_digest(computed_core_sha, report.core_sha256):
        _fail("PROMOTION_REPORT_HASH_MISMATCH", "intrinsic exact core hash")
    if str(core["evaluation_run_id"]) != str(evaluation_run_id):
        _fail("PROMOTION_REPORT_SCHEMA_INVALID", "evaluation run identity")
    ledger = Ledger(Path(ledger_path), Path(workspace_root))
    hard, aggregate = _hard_failures(
        report=report,
        ledger=ledger,
        evaluation_run_id=evaluation_run_id,
        policy=policy,
    )
    inputs = _decision_inputs(
        core, report_core_sha256=report.core_sha256, aggregate=aggregate
    )
    input_sha = sha256_bytes(canonical_json_bytes(inputs))
    if hard:
        return PromotionDecision(
            "reject",
            hard,
            hard,
            (),
            False,
            evaluation_run_id,
            report.core_sha256,
            policy.policy_sha256,
            input_sha,
            inputs,
        )
    soft = _soft_failures(core, policy)
    state = "review_required" if soft else "promote"
    return PromotionDecision(
        state,
        soft,
        (),
        soft,
        True,
        evaluation_run_id,
        report.core_sha256,
        policy.policy_sha256,
        input_sha,
        inputs,
    )


def record_promotion(
    report: ExactReport,
    *,
    policy_path: str | Path,
    ledger_path: str | Path,
    workspace_root: str | Path,
    evaluation_run_id: str,
) -> ExperimentEvent:
    """Re-evaluate immediately, then append one evidence-bound exact decision."""

    decision = evaluate_promotion(
        report,
        policy_path=policy_path,
        ledger_path=ledger_path,
        workspace_root=workspace_root,
        evaluation_run_id=evaluation_run_id,
    )
    policy = load_promotion_policy(policy_path)
    if report.core.get("evidence_kind") != "model_candidate":
        _fail("NON_CANDIDATE_EVIDENCE_KIND", "only learned candidate evidence is decidable")
    payload = exact_promotion_decision_payload(
        evaluation_run_id=evaluation_run_id,
        state=decision.state,
        reason_codes=decision.reason_codes,
        hard_integrity_passed=not decision.hard_failures,
        report_core_sha256=decision.report_core_sha256,
        policy_sha256=decision.policy_sha256,
        decision_input_sha256=decision.decision_input_sha256,
        scorer_lock_sha256=str(report.core["scorer_lock_sha256"]),
        environment_lock_sha256=str(report.core["environment_lock_sha256"]),
        manifest_sha256=str(report.core["manifest_sha256"]),
        members=report.core["members"],
        failed_gate_evidence=_failed_gate_evidence(decision, policy),
    )
    event = ExperimentEvent.create(
        evaluation_run_id, EventType.EXACT_PROMOTION_DECISION, payload
    )
    Ledger(Path(ledger_path), Path(workspace_root)).append(event)
    return event


def record_review_exception(
    *,
    ledger_path: str | Path,
    workspace_root: str | Path,
    evaluation_run_id: str,
    failed_gates: Mapping[str, Any] | None,
    quantitative_tradeoff: str,
    approver: str,
    reason: str,
    downstream_authorization: str,
) -> ExperimentEvent:
    ledger = Ledger(Path(ledger_path), Path(workspace_root))
    evaluations = reconstruct_exact_evaluations(ledger.read_events())
    state = evaluations.get(evaluation_run_id)
    if state is None or state.decision is None:
        _fail("PROMOTION_DECISION_MISSING", evaluation_run_id)
    decision_event = next(
        item
        for item in state.events
        if item.event_type is EventType.EXACT_PROMOTION_DECISION
    )
    decision = state.decision
    expected_failed_gates = decision.get("failed_gate_evidence")
    if not isinstance(expected_failed_gates, Mapping):
        _fail("PROMOTION_EXCEPTION_GATE_MISMATCH", "decision lacks failed-gate evidence")
    if failed_gates is not None and canonical_json_bytes(
        dict(failed_gates)
    ) != canonical_json_bytes(expected_failed_gates):
        _fail("PROMOTION_EXCEPTION_GATE_MISMATCH", "caller-supplied gates drifted")
    payload = exact_promotion_exception_payload(
        evaluation_run_id=evaluation_run_id,
        decision_event_sha256=event_sha256(decision_event),
        failed_gates=expected_failed_gates,
        quantitative_tradeoff=quantitative_tradeoff,
        approver=approver,
        reason=reason,
        downstream_authorization=downstream_authorization,
        report_core_sha256=decision["report_core_sha256"],
        policy_sha256=decision["policy_sha256"],
        decision_input_sha256=decision["decision_input_sha256"],
        scorer_lock_sha256=decision["scorer_lock_sha256"],
        environment_lock_sha256=decision["environment_lock_sha256"],
        manifest_sha256=decision["manifest_sha256"],
    )
    event = ExperimentEvent.create(
        evaluation_run_id, EventType.EXACT_PROMOTION_EXCEPTION, payload
    )
    ledger.append(event)
    return event
