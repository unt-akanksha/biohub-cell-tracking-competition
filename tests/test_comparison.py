from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from biohub_tracker.comparison import ComparisonError, build_paired_comparison
from biohub_tracker.io import canonical_json_bytes, sha256_bytes


POLICY = json.loads(Path("config/evaluation-policy.json").read_text(encoding="utf-8"))
POLICY_SHA = sha256_bytes(canonical_json_bytes(POLICY))


def _text(value):
    result = Decimal(str(value))
    return "0" if result == 0 else format(result.normalize(), "f")


def _members():
    result = []
    for role in ("baseline", "candidate"):
        for fold in ("fold-e1", "fold-e2"):
            token = sha256_bytes(f"{role}:{fold}".encode())
            result.append(
                {
                    "role": role,
                    "fold_id": fold,
                    "producer_run_id": f"{role}-{fold}",
                    "producer_registration_event_sha256": token,
                    "producer_terminal_event_sha256": token,
                    "manifest_sha256": "a" * 64,
                    "train_membership_sha256": token,
                    "calibration_membership_sha256": token,
                    "evaluation_membership_sha256": token,
                    "model_sha256": token,
                    "config_sha256": token,
                    "code_sha256": token,
                    "data_sha256": token,
                    "graph_inventory_sha256": token,
                    "artifact_hashes": {"model": token, "predictions": token},
                }
            )
    return result


def _diagnostic(tp, fn):
    denominator = tp + fn
    value = _text(Decimal(tp) / Decimal(denominator)) if denominator else None
    status = "applicable" if denominator else "not_applicable"
    scalar = {
        "status": status,
        "reason": None if denominator else "no_gt_edges",
        "numerator": tp,
        "denominator": denominator,
        "value": value,
    }
    return {
        "authority": "non_authoritative_diagnostic",
        "organizer_input_eligible": False,
        "endpoint_availability": scalar,
        "conditional_association_recall": scalar,
        "conditional_valid_edge_precision": scalar,
        "conditional_valid_edge_jaccard": scalar,
        "oracle_link_ceiling": {
            "raw": scalar,
            "adjusted": {
                "status": status,
                "reason": None if denominator else "no_gt_edges",
                "value": value,
            },
        },
        "oracle_gap_adjusted_edge": {
            "status": status,
            "reason": None if denominator else "no_gt_edges",
            "value": "0" if denominator else None,
        },
        "node_count_ratio": "0",
    }


def _movie(sample_id, embryo, fold, producer, tp, fp, fn):
    denominator = tp + fp + fn
    edge = Decimal(tp) / Decimal(denominator) if denominator else Decimal(0)
    return {
        "sample_id": sample_id,
        "embryo_id": embryo,
        "fold_id": fold,
        "producer": producer,
        "organizer_row": {"bound": True},
        "diagnostic_state": _diagnostic(tp, fn),
        "official_counts": {
            "edge_tp": tp,
            "edge_fp": fp,
            "edge_fn": fn,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
            "num_pred_nodes": 10,
        },
        "estimated_number_of_nodes": "10",
        "gt_node_count": 10,
        "matched_gt_node_count": 8 + int(tp > fn),
        "node_recall": _text(Decimal(8 + int(tp > fn)) / Decimal(10)),
        "node_count_ratio": "0",
        "edge_jaccard": _text(edge),
        "adjusted_edge_jaccard": _text(edge),
        "division_jaccard": None,
        "score": _text(edge),
    }


def _aggregate(rows):
    counts = {
        name: sum(int(row["official_counts"][name]) for row in rows)
        for name in rows[0]["official_counts"]
    }
    denominator = counts["edge_tp"] + counts["edge_fp"] + counts["edge_fn"]
    edge = Decimal(counts["edge_tp"]) / Decimal(denominator)
    weight = sum(
        row["official_counts"]["edge_tp"]
        + row["official_counts"]["edge_fp"]
        + row["official_counts"]["edge_fn"]
        for row in rows
    )
    adjusted = sum(
        Decimal(row["adjusted_edge_jaccard"])
        * (
            row["official_counts"]["edge_tp"]
            + row["official_counts"]["edge_fp"]
            + row["official_counts"]["edge_fn"]
        )
        for row in rows
    ) / Decimal(weight)
    gt_nodes = sum(row["gt_node_count"] for row in rows)
    matched = sum(row["matched_gt_node_count"] for row in rows)
    return {
        "movie_count": len(rows),
        "official_counts": counts,
        "estimated_number_of_nodes": _text(
            sum(Decimal(row["estimated_number_of_nodes"]) for row in rows)
        ),
        "gt_node_count": gt_nodes,
        "matched_gt_node_count": matched,
        "adjusted_edge_weight": weight,
        "edge_jaccard": _text(edge),
        "adjusted_edge_jaccard": _text(adjusted),
        "division_jaccard": None,
        "organizer_macro_node_recall": _text(
            sum(Decimal(row["node_recall"]) for row in rows) / Decimal(len(rows))
        ),
        "node_recall_micro": _text(Decimal(matched) / Decimal(gt_nodes)),
        "score": _text(adjusted),
    }


def _fixture():
    members = _members()
    by_slot = {(item["role"], item["fold_id"]): item for item in members}
    specifications = (
        ("e1-a", "e1", "fold-e1", 5, 1, 4, 6, 1, 3),
        ("e1-b", "e1", "fold-e1", 7, 1, 2, 6, 1, 3),
        ("e2-a", "e2", "fold-e2", 4, 2, 4, 7, 1, 2),
        ("e2-b", "e2", "fold-e2", 8, 1, 1, 8, 1, 1),
    )
    baseline = []
    candidate = []
    for sample_id, embryo, fold, btp, bfp, bfn, ctp, cfp, cfn in specifications:
        baseline.append(
            _movie(sample_id, embryo, fold, by_slot[("baseline", fold)], btp, bfp, bfn)
        )
        candidate.append(
            _movie(sample_id, embryo, fold, by_slot[("candidate", fold)], ctp, cfp, cfn)
        )
    inventories = [
        {
            "role": member["role"],
            "fold_id": member["fold_id"],
            "producer_run_id": member["producer_run_id"],
            "submission_graph_inventory_sha256": member["graph_inventory_sha256"],
            "roundtrip_evidence_sha256": sha256_bytes(
                f"roundtrip:{member['producer_run_id']}".encode()
            ),
            "csv_sha256": sha256_bytes(f"csv:{member['producer_run_id']}".encode()),
        }
        for member in members
    ]
    registration = {
        "evaluation_run_id": "eval",
        "scorer_lock_sha256": "b" * 64,
        "environment_lock_sha256": "c" * 64,
        "manifest_sha256": "a" * 64,
        "evaluation_policy_sha256": POLICY_SHA,
        "members": members,
    }
    return registration, members, inventories, baseline, candidate


def _build(**overrides):
    registration, members, inventories, baseline, candidate = _fixture()
    values = {
        "registration": registration,
        "aggregate_status": "running",
        "members": members,
        "authoritative_inventories": inventories,
        "baseline_movies": baseline,
        "candidate_movies": candidate,
        "expected_sample_ids": [row["sample_id"] for row in baseline],
        "policy": POLICY,
        "policy_sha256": POLICY_SHA,
        "aggregate": _aggregate,
    }
    values.update(overrides)
    return build_paired_comparison(**values)


def test_bootstrap_is_byte_deterministic_paired_and_movie_stratified():
    first = _build()
    second = _build()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["bootstrap"]["seed"] == 20260824
    assert first["bootstrap"]["repetitions"] == 10000
    assert first["bootstrap"]["unit"] == "complete_movie_within_embryo"
    assert first["bootstrap"]["no_division_replicate_count"] == 10000
    assert first["bootstrap"]["metrics"]["score"]["valid_replicates"] == 10000
    assert first["bootstrap"]["metrics"]["division_jaccard"]["status"] == "not_applicable"
    assert len(first["by_movie"]) == 4
    assert first["worst_paired_movie"]["sample_id"] == "e1-b"


def test_comparison_boundary_rejects_lifecycle_and_member_drift_before_aggregation():
    calls = 0

    def aggregate(rows):
        nonlocal calls
        calls += 1
        return _aggregate(rows)

    with pytest.raises(ComparisonError, match="aggregate lifecycle"):
        _build(aggregate_status="completed", aggregate=aggregate)
    assert calls == 0

    registration, members, inventories, baseline, candidate = _fixture()
    candidate = deepcopy(candidate)
    candidate[0]["producer"]["producer_terminal_event_sha256"] = "f" * 64
    with pytest.raises(ComparisonError, match="producer binding"):
        _build(
            registration=registration,
            members=members,
            authoritative_inventories=inventories,
            baseline_movies=baseline,
            candidate_movies=candidate,
            aggregate=aggregate,
        )
    assert calls == 0


def test_canonical_comparison_contains_complete_deltas_and_no_public_score():
    result = _build()
    assert set(result["by_embryo"]) == {"e1", "e2"}
    assert set(result["by_fold"]) == {"fold-e1", "fold-e2"}
    assert result["pooled"]["metrics"]["score"]["delta"] is not None
    assert result["pooled"]["counts"]["edge_tp"]["delta"] == 3
    required_diagnostics = {
        "conditional_association_recall",
        "conditional_valid_edge_precision",
        "conditional_valid_edge_jaccard",
        "endpoint_availability",
        "node_count_ratio",
        "oracle_gap_adjusted_edge",
        "oracle_link_ceiling_adjusted",
        "oracle_link_ceiling_raw",
    }
    assert required_diagnostics <= set(result["pooled"]["diagnostic_deltas"])
    for group in (result["by_embryo"], result["by_fold"]):
        for row in group.values():
            assert required_diagnostics <= set(row["diagnostic_deltas"])
    for row in result["by_movie"]:
        assert required_diagnostics <= set(row["diagnostic_deltas"])
    assert result["lowest_absolute_candidate_movie"]["sample_id"] == "e1-a"
    serialized = canonical_json_bytes(result).decode("utf-8")
    assert "public_score" not in serialized
    assert "leaderboard" not in serialized


def test_diagnostic_pooling_recomputes_ratios_node_counts_and_oracle_gap():
    registration, members, inventories, baseline, candidate = _fixture()
    baseline = deepcopy(baseline)
    candidate = deepcopy(candidate)
    baseline[0]["official_counts"]["num_pred_nodes"] = 20
    baseline[0]["diagnostic_state"]["node_count_ratio"] = "1"
    baseline[1]["official_counts"]["num_pred_nodes"] = 5
    baseline[1]["diagnostic_state"]["node_count_ratio"] = "-0.5"
    result = _build(
        registration=registration,
        members=members,
        authoritative_inventories=inventories,
        baseline_movies=baseline,
        candidate_movies=candidate,
    )
    pooled = result["pooled"]["diagnostic_deltas"]
    assert pooled["node_count_ratio"]["baseline"] == "0.125"
    assert pooled["node_count_ratio"]["baseline"] != "0.25"
    assert pooled["conditional_association_recall"]["baseline"] == _text(
        Decimal(24) / Decimal(35)
    )
    assert pooled["oracle_link_ceiling_adjusted"]["baseline"] == _text(
        Decimal(24) / Decimal(35) * Decimal("0.9875")
    )
    assert pooled["oracle_gap_adjusted_edge"]["baseline"] is not None


def test_comparison_rejects_missing_required_diagnostic_metric():
    registration, members, inventories, baseline, candidate = _fixture()
    candidate = deepcopy(candidate)
    del candidate[0]["diagnostic_state"]["conditional_valid_edge_jaccard"]
    with pytest.raises(ComparisonError, match="required diagnostic field"):
        _build(
            registration=registration,
            members=members,
            authoritative_inventories=inventories,
            baseline_movies=baseline,
            candidate_movies=candidate,
        )
