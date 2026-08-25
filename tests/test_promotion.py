from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from biohub_tracker.evaluation import ExactReport, validate_exact_core
from biohub_tracker.io import canonical_json_bytes, sha256_bytes
from biohub_tracker.ledger import (
    EventType,
    ExperimentEvent,
    Ledger,
    TransitionError,
    completed_payload,
    event_sha256,
    exact_evaluation_completed_payload,
    exact_evaluation_registration_payload,
    exact_evaluation_started_payload,
    registration_payload,
    resolved_exact_member,
    start_payload,
)
from biohub_tracker.promotion import (
    PromotionError,
    evaluate_promotion,
    load_promotion_policy,
    record_promotion,
    record_review_exception,
)


POLICY = Path("config/promotion-policy.json").resolve()
FOLDS = ("fold-44b6-to-6bba", "fold-6bba-to-44b6")


def _legacy_metrics() -> dict:
    return {
        "pooled": {
            "adjusted_edge_jaccard": 1,
            "edge_jaccard": 1,
            "division_jaccard": 1,
            "node_recall": 1,
        },
        "division_counts": {"tp": 1, "fp": 0, "fn": 0},
        "by_embryo": {"44b6": {}, "6bba": {}},
        "by_fold": {"fold": {}},
        "worst_movie_delta": 0,
    }


def _metric_delta(delta: str = "0.001") -> dict[str, object]:
    return {
        "status": "applicable",
        "reason": None,
        "baseline": "0.9",
        "candidate": str(0.9 + float(delta)),
        "delta": delta,
    }


def _group(delta: str = "0.001") -> dict[str, object]:
    return {
        "metrics": {
            "score": _metric_delta(delta),
            "adjusted_edge_jaccard": _metric_delta(delta),
            "edge_jaccard": _metric_delta(delta),
            "division_jaccard": _metric_delta("0"),
            "organizer_macro_node_recall": _metric_delta("0"),
            "node_recall_micro": _metric_delta("0"),
        },
        "counts": {
            "division_tp": {"baseline": 1, "candidate": 1, "delta": 0},
        },
    }


def _producer(ledger: Ledger, role: str, fold_id: str, token: str) -> dict:
    run_id = f"{role}-{token}"
    config = {"role": role, "token": token}
    base = registration_payload(
        hypothesis=run_id,
        parent=None,
        config=config,
        seeds=[1],
        split=fold_id,
        declared_max_runtime_hours="1",
        code={"git_head": token},
    )
    lineage = {
        "manifest_sha256": "1" * 64,
        "fold_id": fold_id,
        "train_membership_sha256": token * 64,
        "calibration_membership_sha256": token * 64,
        "evaluation_membership_sha256": token * 64,
        "model_sha256": token * 64,
        "config_sha256": base["config_sha256"],
        "code_sha256": token * 64,
        "data_sha256": token * 64,
    }
    ledger.append(
        ExperimentEvent.create(
            run_id,
            EventType.REGISTERED,
            registration_payload(
                hypothesis=run_id,
                parent=None,
                config=config,
                seeds=[1],
                split=fold_id,
                declared_max_runtime_hours="1",
                code={"git_head": token},
                producer_evidence=lineage,
            ),
        )
    )
    ledger.append(
        ExperimentEvent.create(
            run_id,
            EventType.STARTED,
            start_payload(
                kaggle_ref="owner/kernel",
                authorization_id=run_id,
                quota_before_hours="30",
            ),
        )
    )
    ledger.append(
        ExperimentEvent.create(
            run_id,
            EventType.COMPLETED,
            completed_payload(
                actual_runtime_hours="1",
                quota_after_hours="29",
                metrics=_legacy_metrics(),
                producer_evidence={
                    "evidence_eligible": True,
                    "graph_inventory_sha256": token * 64,
                    "artifact_hashes": {"predictions": token * 64},
                },
            ),
        )
    )
    return resolved_exact_member(
        ledger.read_events(), role=role, fold_id=fold_id, producer_run_id=run_id
    )


def _completed_report(
    tmp_path: Path,
    *,
    evidence_kind: str = "model_candidate",
    integrity_override: dict[str, object] | None = None,
) -> tuple[Ledger, ExactReport]:
    ledger = Ledger(tmp_path / "experiments/events.jsonl", tmp_path)
    members = []
    for index, role in enumerate(("baseline", "candidate")):
        for offset, fold in enumerate(FOLDS):
            members.append(_producer(ledger, role, fold, "2345"[index * 2 + offset]))
    members.sort(key=lambda item: (item["role"], item["fold_id"]))
    registration = exact_evaluation_registration_payload(
        evaluation_run_id="evaluation-candidate-v1",
        scorer_lock_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
        manifest_sha256="1" * 64,
        evaluation_policy_sha256="c" * 64,
        evidence_kind=evidence_kind,
        members=members,
    )
    ledger.append(
        ExperimentEvent.create(
            "evaluation-candidate-v1",
            EventType.EXACT_EVALUATION_REGISTERED,
            registration,
        )
    )
    ledger.append(
        ExperimentEvent.create(
            "evaluation-candidate-v1",
            EventType.EXACT_EVALUATION_STARTED,
            exact_evaluation_started_payload(evaluation_run_id="evaluation-candidate-v1"),
        )
    )
    inventories = [
        {
            "role": member["role"],
            "fold_id": member["fold_id"],
            "producer_run_id": member["producer_run_id"],
            "submission_graph_inventory_sha256": member["graph_inventory_sha256"],
            "roundtrip_evidence_sha256": member["model_sha256"],
            "csv_sha256": member["data_sha256"],
        }
        for member in members
    ]
    integrity = {
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
    integrity.update(integrity_override or {})
    core = {
        "schema_version": "biohub.exact-report.v2",
        "evaluation_run_id": "evaluation-candidate-v1",
        "evidence_kind": evidence_kind,
        "scorer_lock_sha256": "a" * 64,
        "environment_lock_sha256": "b" * 64,
        "manifest_sha256": "1" * 64,
        "evaluation_policy_sha256": "c" * 64,
        "members": members,
        "authoritative_inventories": inventories,
        "coverage": {
            "expected_sample_ids": ["44b6_a", "6bba_a"],
            "baseline_sample_ids": ["44b6_a", "6bba_a"],
            "candidate_sample_ids": ["44b6_a", "6bba_a"],
            "missing": [],
            "extra": [],
            "complete": True,
        },
        "official": {"baseline": {}, "candidate": {}},
        "diagnostics": {
            "authority": "non_authoritative_diagnostic",
            "organizer_input_eligible": False,
            "baseline": {},
            "candidate": {},
        },
        "comparison": {
            "schema_version": "biohub.exact-comparison.v1",
            "direction": "candidate_minus_baseline",
            "pooled": _group(),
            "by_embryo": {"44b6": _group(), "6bba": _group()},
            "by_fold": {fold: _group() for fold in FOLDS},
            "by_movie": [],
            "worst_paired_movie": {
                "sample_id": "44b6_a",
                "metrics": {"score": _metric_delta("-0.001")},
            },
            "lowest_absolute_candidate_movie": {},
            "bootstrap": {
                "metrics": {
                    "score": {
                        "status": "applicable",
                        "valid_replicates": 10000,
                        "percentiles": {"2.5": "0.0001", "50": "0.001", "97.5": "0.002"},
                        "probability_candidate_gt_baseline": "0.75",
                    }
                }
            },
            "integrity_checks": {
                "aggregate_lifecycle": "running_then_completed_on_report_attachment"
            },
        },
        "integrity_checks": integrity,
        "promotion_eligible": False,
    }
    core_sha = sha256_bytes(canonical_json_bytes(core))
    envelope = {
        "schema_version": "biohub.exact-report-envelope.v1",
        "report_core_sha256": core_sha,
        "created_at": "2026-08-24T00:00:00Z",
        "runtime_seconds": "1",
        "peak_memory_bytes": 1,
        "presentation_metadata": {"public_score": "0.999"},
    }
    envelope_sha = sha256_bytes(canonical_json_bytes(envelope))
    ledger.append(
        ExperimentEvent.create(
            "evaluation-candidate-v1",
            EventType.EXACT_EVALUATION_COMPLETED,
            exact_evaluation_completed_payload(
                evaluation_run_id="evaluation-candidate-v1",
                members=members,
                report_core_sha256=core_sha,
                envelope_sha256=envelope_sha,
                artifact_hashes={"core": core_sha, "envelope": envelope_sha},
                authoritative_inventories=inventories,
                promotion_eligible=evidence_kind == "model_candidate",
            ),
        )
    )
    return ledger, ExactReport(core, envelope, core_sha, envelope_sha)


def _evaluate(ledger: Ledger, report: ExactReport):
    return evaluate_promotion(
        report,
        policy_path=POLICY,
        ledger_path=ledger.path,
        workspace_root=ledger.workspace_root,
        evaluation_run_id="evaluation-candidate-v1",
    )


def _replace_core_and_attachment(ledger: Ledger, report: ExactReport, core: dict) -> ExactReport:
    core_sha = sha256_bytes(canonical_json_bytes(core))
    rewritten = []
    for event in ledger.read_events():
        if event.event_type is EventType.EXACT_EVALUATION_COMPLETED:
            payload = dict(event.payload)
            payload["report_core_sha256"] = core_sha
            payload["artifact_hashes"] = {
                **payload["artifact_hashes"],
                "core": core_sha,
            }
            event = ExperimentEvent.create(
                event.run_id,
                event.event_type,
                payload,
                created_at=event.created_at,
                event_id=event.event_id,
            )
        rewritten.append(event)
    ledger.path.write_bytes(
        b"".join(canonical_json_bytes(event.to_dict()) + b"\n" for event in rewritten)
    )
    return replace(report, core=core, core_sha256=core_sha)


def test_reject_integrity_gates_are_hard_and_deterministically_ordered(tmp_path):
    ledger, report = _completed_report(
        tmp_path,
        integrity_override={
            "complete_coverage": "failed",
            "metric_exploit_audit": "failed",
        },
    )
    decision = _evaluate(ledger, report)
    assert decision.state == "reject"
    assert decision.reason_codes == (
        "METRIC_EXPLOIT_SIGNATURE",
        "INCOMPLETE_MOVIE_COVERAGE",
    )
    assert decision.soft_gates_evaluated is False


def test_missing_metric_exploit_audit_is_rejected_but_legacy_v1_remains_readable(tmp_path):
    ledger, report = _completed_report(tmp_path)
    missing = dict(report.core)
    missing["integrity_checks"] = dict(missing["integrity_checks"])
    missing["integrity_checks"].pop("metric_exploit_audit")
    rewritten = _replace_core_and_attachment(ledger, report, missing)
    with pytest.raises(PromotionError, match="metric exploit audit is required"):
        _evaluate(ledger, rewritten)

    legacy = dict(missing)
    legacy["schema_version"] = "biohub.exact-report.v1"
    assert validate_exact_core(legacy)["schema_version"] == "biohub.exact-report.v1"


@pytest.mark.parametrize("kind", ["synthetic_fixture", "official_data_control"])
def test_reject_non_candidate_evidence_kind_before_soft_gates(tmp_path, kind):
    ledger, report = _completed_report(tmp_path, evidence_kind=kind)
    decision = _evaluate(ledger, report)
    assert decision.state == "reject"
    assert decision.reason_codes == ("NON_CANDIDATE_EVIDENCE_KIND",)
    assert decision.soft_gates_evaluated is False


def test_public_score_add_change_or_delete_does_not_change_decision_bytes(tmp_path):
    ledger, report = _completed_report(tmp_path)
    first = _evaluate(ledger, report)
    changed_envelope = dict(report.envelope)
    changed_envelope["presentation_metadata"] = {
        "public_score": "0.001",
        "leaderboard_score": "9.99",
    }
    changed = replace(
        report,
        envelope=changed_envelope,
        envelope_sha256=sha256_bytes(canonical_json_bytes(changed_envelope)),
    )
    second = _evaluate(ledger, changed)
    deleted_envelope = dict(report.envelope)
    deleted_envelope["presentation_metadata"] = {}
    deleted = replace(
        report,
        envelope=deleted_envelope,
        envelope_sha256=sha256_bytes(canonical_json_bytes(deleted_envelope)),
    )
    third = _evaluate(ledger, deleted)
    assert first.to_dict() == second.to_dict() == third.to_dict()
    assert b"public" not in canonical_json_bytes(first.decision_inputs).lower()


def test_reject_unknown_or_nonterminal_producer_from_stale_aggregate(tmp_path):
    ledger, report = _completed_report(tmp_path)
    events = ledger.read_events()
    unknown_id = report.core["members"][0]["producer_run_id"]
    retained = [event for event in events if event.run_id != unknown_id]
    ledger.path.write_bytes(b"".join(canonical_json_bytes(event.to_dict()) + b"\n" for event in retained))
    unknown = _evaluate(ledger, report)
    assert unknown.state == "reject"
    assert unknown.reason_codes == ("UNKNOWN_PRODUCER",)

    ledger, report = _completed_report(tmp_path / "nonterminal")
    events = ledger.read_events()
    target = report.core["members"][0]["producer_run_id"]
    retained = [
        event
        for event in events
        if not (event.run_id == target and event.event_type is EventType.COMPLETED)
    ]
    ledger.path.write_bytes(b"".join(canonical_json_bytes(event.to_dict()) + b"\n" for event in retained))
    nonterminal = _evaluate(ledger, report)
    assert nonterminal.state == "reject"
    assert nonterminal.reason_codes == ("NONTERMINAL_PRODUCER",)


def test_reject_registered_and_terminal_hash_drift_before_thresholds(tmp_path):
    ledger, report = _completed_report(tmp_path)
    events = ledger.read_events()
    member = report.core["members"][0]
    rewritten = []
    for event in events:
        if event.run_id == member["producer_run_id"] and event.event_type is EventType.REGISTERED:
            payload = dict(event.payload)
            payload["model_sha256"] = "f" * 64
            event = ExperimentEvent.create(
                event.run_id,
                event.event_type,
                payload,
                created_at=event.created_at,
                event_id=event.event_id,
            )
        rewritten.append(event)
    ledger.path.write_bytes(b"".join(canonical_json_bytes(event.to_dict()) + b"\n" for event in rewritten))
    decision = _evaluate(ledger, report)
    assert decision.state == "reject"
    assert decision.reason_codes == ("REGISTERED_LINEAGE_MISMATCH",)


def test_reject_incomplete_or_report_mismatched_aggregate(tmp_path):
    ledger, report = _completed_report(tmp_path)
    events = [
        event
        for event in ledger.read_events()
        if event.event_type is not EventType.EXACT_EVALUATION_COMPLETED
    ]
    ledger.path.write_bytes(b"".join(canonical_json_bytes(event.to_dict()) + b"\n" for event in events))
    decision = _evaluate(ledger, report)
    assert decision.state == "reject"
    assert decision.reason_codes == ("EXACT_EVALUATION_NOT_COMPLETED",)


def test_reject_policy_and_report_schema_or_hash_errors_without_decision(tmp_path):
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy["unknown"] = True
    bad_policy = tmp_path / "bad-policy.json"
    bad_policy.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(PromotionError, match="PROMOTION_POLICY_SCHEMA_INVALID"):
        load_promotion_policy(bad_policy)

    ledger, report = _completed_report(tmp_path / "report")
    bad_core = dict(report.core)
    bad_core["unknown"] = float("nan")
    with pytest.raises(PromotionError):
        _evaluate(ledger, replace(report, core=bad_core))


def test_promotion_and_review_required_threshold_boundary_are_frozen(tmp_path):
    ledger, report = _completed_report(tmp_path)
    assert _evaluate(ledger, report).state == "promote"
    core = dict(report.core)
    comparison = dict(core["comparison"])
    pooled = dict(comparison["pooled"])
    metrics = dict(pooled["metrics"])
    metrics["score"] = _metric_delta("0.00000")
    pooled["metrics"] = metrics
    comparison["pooled"] = pooled
    core["comparison"] = comparison
    boundary = _replace_core_and_attachment(ledger, report, core)
    decision = _evaluate(ledger, boundary)
    assert decision.state == "review_required"
    assert decision.reason_codes == ("POOLED_SCORE_NOT_POSITIVE",)
    assert decision.hard_failures == ()


def test_record_decision_and_review_exception_are_immutable_and_evidence_bound(tmp_path):
    ledger, report = _completed_report(tmp_path)
    core = dict(report.core)
    comparison = dict(core["comparison"])
    pooled = dict(comparison["pooled"])
    metrics = dict(pooled["metrics"])
    metrics["node_recall_micro"] = _metric_delta("-0.00001")
    pooled["metrics"] = metrics
    comparison["pooled"] = pooled
    core["comparison"] = comparison
    report = _replace_core_and_attachment(ledger, report, core)
    decision_event = record_promotion(
        report,
        policy_path=POLICY,
        ledger_path=ledger.path,
        workspace_root=ledger.workspace_root,
        evaluation_run_id="evaluation-candidate-v1",
    )
    assert decision_event.payload["state"] == "review_required"
    exception = record_review_exception(
        ledger_path=ledger.path,
        workspace_root=ledger.workspace_root,
        evaluation_run_id="evaluation-candidate-v1",
        failed_gates={"NODE_RECALL_REGRESSION": {"delta": "-0.00001"}},
        quantitative_tradeoff="score gain exceeds the small node-recall regression",
        approver="competition-owner",
        reason="explicitly reviewed for the next controlled experiment only",
        downstream_authorization="phase3-experiment-only",
    )
    assert exception.payload["decision_event_sha256"] == event_sha256(decision_event)
    before = ledger.path.read_bytes()
    with pytest.raises(TransitionError):
        ledger.append(exception)
    assert ledger.path.read_bytes() == before


def test_review_exception_missing_required_audit_fields_fails_without_append(tmp_path):
    ledger, report = _completed_report(tmp_path)
    record_promotion(
        report,
        policy_path=POLICY,
        ledger_path=ledger.path,
        workspace_root=ledger.workspace_root,
        evaluation_run_id="evaluation-candidate-v1",
    )
    before = ledger.path.read_bytes()
    with pytest.raises(PromotionError, match="PROMOTION_DECISION_MISSING"):
        record_review_exception(
            ledger_path=ledger.path,
            workspace_root=ledger.workspace_root,
            evaluation_run_id="missing",
            failed_gates={},
            quantitative_tradeoff="x",
            approver="x",
            reason="x",
            downstream_authorization="x",
        )
    assert ledger.path.read_bytes() == before
