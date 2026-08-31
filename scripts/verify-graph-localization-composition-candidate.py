#!/usr/bin/env python
"""Verify the independently promoted graph plus localization composition."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GRAPH = runpy.run_path(str(ROOT / "scripts/verify-graph-context-consensus-submission-candidate.py"))
LOCALIZATION = runpy.run_path(str(ROOT / "scripts/verify-temporal-localization-submission-candidate.py"))
COMMON = runpy.run_path(str(ROOT / "scripts/verify-learned-division-submission-candidate.py"))
RANKED = runpy.run_path(str(ROOT / "scripts/verify-ranked-consensus-submission-candidate.py"))

aggregate_validator = COMMON["aggregate_validator"]
read_csv_rows = COMMON["read_csv_rows"]
sha256_file = COMMON["sha256_file"]
unique_file = COMMON["unique_file"]
validate_submission_csv = COMMON["validate_submission_csv"]
assert_close = COMMON["_assert_close"]
sum_integral = RANKED["_sum_integral"]
validator_node_errors = LOCALIZATION["validator_node_errors"]

RUN_ID = "ema-graph-localization-composition-v1"
TARGET_PUBLIC_SCORE = 0.945
GRAPH_RUN_ID = GRAPH["RUN_ID"]
LOCALIZATION_RUN_ID = LOCALIZATION["RUN_ID"]
GRAPH_PARAMETER_COUNT = GRAPH["EXPECTED_PARAMETER_COUNT"]
LOCALIZATION_PARAMETER_COUNT = LOCALIZATION["EXPECTED_PARAMETER_COUNT"]
KNOWN_PUBLIC_SUBMISSION_SHA256 = COMMON["KNOWN_PUBLIC_SUBMISSION_SHA256"]
PUBLIC_CONTROL_VALIDATOR_SHA256 = COMMON["PUBLIC_CONTROL_VALIDATOR_SHA256"]
MINIMUM_PROXY_GAIN = 0.005
MAXIMUM_ADJUSTED_EDGE_REGRESSION = 0.001
MINIMUM_COMPOSITION_GAIN_OVER_BEST_COMPONENT = 0.001


def validate_component_promotion(
    path: Path,
    *,
    component: str,
    expected_run_id: str,
    expected_runtime_sha256: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    validator = report.get("candidate_validator", {})
    if not (
        report.get("schema_version") == 1
        and report.get("status") == "eligible_for_submission"
        and report.get("run_id") == expected_run_id
        and report.get("target_public_score") == TARGET_PUBLIC_SCORE
        and report.get("runtime_manifest_sha256") == expected_runtime_sha256
        and report.get("known_public_hash_match") is False
        and report.get("competition_submission_performed") is False
        and report.get("authorized_for_submission") is True
        and isinstance(report.get("submission_sha256"), str)
        and len(report["submission_sha256"]) == 64
        and isinstance(validator, dict)
        and float(validator.get("proxy_score", -math.inf)) > 0.0
    ):
        raise RuntimeError(f"{component} standalone promotion report is invalid")
    return report


def _assert_evidence_metrics(
    evidence: dict[str, Any],
    observed: dict[str, int | float],
) -> None:
    for key, value in observed.items():
        assert_close(key, evidence.get(key), float(value))


def verify_candidate(
    output_root: Path,
    baseline_validator: Path,
    graph_runtime_manifest: Path,
    localization_runtime_manifest: Path,
    graph_promotion_report: Path,
    localization_promotion_report: Path,
    *,
    expected_baseline_sha256: str | None = None,
) -> dict[str, Any]:
    graph_runtime = GRAPH["validate_runtime"](graph_runtime_manifest)
    localization_runtime = LOCALIZATION["validate_runtime"](localization_runtime_manifest)
    graph_promotion = validate_component_promotion(
        graph_promotion_report,
        component="graph-context",
        expected_run_id=GRAPH_RUN_ID,
        expected_runtime_sha256=graph_runtime["manifest_sha256"],
    )
    localization_promotion = validate_component_promotion(
        localization_promotion_report,
        component="temporal-localization",
        expected_run_id=LOCALIZATION_RUN_ID,
        expected_runtime_sha256=localization_runtime["manifest_sha256"],
    )
    baseline_sha256 = sha256_file(baseline_validator)
    if expected_baseline_sha256 is not None and baseline_sha256 != expected_baseline_sha256:
        raise RuntimeError("public-control validator artifact changed")

    terminal_path = unique_file(output_root, "watchdog-terminal.json")
    evidence_path = unique_file(output_root, "candidate_evidence.json")
    submission_path = unique_file(output_root, "submission.csv")
    run_stats_path = unique_file(output_root, "run_stats.csv")
    validator_path = unique_file(output_root, "validator_results.csv")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    submission_sha256 = sha256_file(submission_path)
    if not (
        terminal.get("run_id") == RUN_ID
        and terminal.get("status") == "completed"
        and terminal.get("submission_exists") is True
        and terminal.get("submission_sha256") == submission_sha256
        and terminal.get("declared_budget_seconds") == 39_600
        and terminal.get("safety_margin_seconds") == 1_200
        and float(terminal.get("elapsed_seconds", math.inf)) < 39_600.0
    ):
        raise RuntimeError("composition watchdog terminal is invalid")

    graph_hashes = graph_runtime["deep_model_sha256"]
    localization_hashes = localization_runtime["model_sha256"]
    if not (
        evidence.get("schema_version") == 1
        and evidence.get("status") == "completed_pending_external_promotion_gate"
        and evidence.get("run_id") == RUN_ID
        and evidence.get("target_public_score") == TARGET_PUBLIC_SCORE
        and evidence.get("public_lineage_attributed") is True
        and evidence.get("public_predictions_copied") is False
        and evidence.get("component_order")
        == ["graph_context_division", "temporal_localization"]
        and evidence.get("independent_component_promotion_required") is True
        and evidence.get("graph_runtime_manifest_sha256")
        == graph_runtime["manifest_sha256"]
        and evidence.get("graph_policy") == graph_runtime["deep_policy"]
        and evidence.get("graph_member_count") == graph_runtime["deep_member_count"]
        and evidence.get("graph_parameters_per_member") == GRAPH_PARAMETER_COUNT
        and evidence.get("graph_model_sha256") == graph_hashes
        and evidence.get("morphology_model_sha256")
        == graph_runtime["morphology_model_sha256"]
        and evidence.get("localization_runtime_manifest_sha256")
        == localization_runtime["manifest_sha256"]
        and evidence.get("localization_member_count")
        == localization_runtime["member_count"]
        and evidence.get("localization_parameters_per_member")
        == LOCALIZATION_PARAMETER_COUNT
        and evidence.get("localization_model_sha256") == localization_hashes
        and evidence.get("absolute_threshold_used") is False
        and evidence.get("node_count_preserving") is True
        and evidence.get("model_subset_searched_on_audit") is False
        and evidence.get("weights_searched_on_development") is False
        and evidence.get("threshold_searched_on_development") is False
        and evidence.get("competition_submission_performed") is False
        and evidence.get("authorized_for_submission") is False
        and evidence.get("submission_sha256") == submission_sha256
    ):
        raise RuntimeError("composition candidate evidence is invalid")
    if submission_sha256 in KNOWN_PUBLIC_SUBMISSION_SHA256:
        raise RuntimeError("composition is identical to an audited public output")

    submission = validate_submission_csv(submission_path)
    run_rows = read_csv_rows(run_stats_path)
    if not run_rows:
        raise RuntimeError("composition run stats are empty")

    geometric = sum_integral(run_rows, "ranked_consensus_geometric_candidates")
    eligible = sum_integral(run_rows, "ranked_consensus_geometry_eligible_candidates")
    scored = sum_integral(run_rows, "ranked_consensus_candidate_parents_scored")
    agreements = sum_integral(run_rows, "ranked_consensus_ranking_agreed")
    added = sum_integral(run_rows, "ranked_consensus_added_edges")
    maximum_additions = sum_integral(run_rows, "ranked_consensus_maximum_additions")
    absolute_threshold = sum_integral(run_rows, "ranked_consensus_absolute_threshold_used")
    reassignments = sum_integral(run_rows, "ranked_consensus_reassignment_performed")
    graph_node_changes = sum_integral(run_rows, "ranked_consensus_node_or_coordinate_changes")
    graph_gpu_groups = sum_integral(run_rows, "ranked_consensus_gpu_groups_used")
    if not (
        geometric > 0
        and 0 < eligible <= geometric
        and scored == eligible
        and 0 < agreements == added <= maximum_additions
        and absolute_threshold == 0
        and reassignments == 0
        and graph_node_changes == 0
        and graph_gpu_groups
        == len(run_rows) * graph_runtime["expected_gpu_groups"]
    ):
        raise RuntimeError("composition graph-context integrity gate failed")

    localization_candidates = sum_integral(
        run_rows, "ranked_consensus_localization_candidate_nodes"
    )
    moved = sum_integral(run_rows, "ranked_consensus_localization_nodes_moved")
    rounded_changes = sum_integral(
        run_rows, "ranked_consensus_localization_rounded_coordinate_changes"
    )
    boundary_rejected = sum_integral(
        run_rows, "ranked_consensus_localization_boundary_rejected"
    )
    global_failures = sum_integral(
        run_rows, "ranked_consensus_localization_global_gate_failures"
    )
    localization_node_changes = sum_integral(
        run_rows, "ranked_consensus_localization_node_count_changes"
    )
    localization_edge_changes = sum_integral(
        run_rows, "ranked_consensus_localization_edge_changes"
    )
    localization_gpu_groups = sum_integral(
        run_rows, "ranked_consensus_localization_gpu_groups_used"
    )
    if not (
        localization_candidates > 0
        and 0 < rounded_changes <= moved <= localization_candidates
        and 0 <= boundary_rejected <= localization_candidates
        and global_failures == 0
        and localization_node_changes == 0
        and localization_edge_changes == 0
        and localization_gpu_groups
        == len(run_rows) * localization_runtime["expected_gpu_groups"]
    ):
        raise RuntimeError("composition temporal-localization integrity gate failed")

    _assert_evidence_metrics(
        evidence,
        {
            "ranked_geometric_candidates": geometric,
            "ranked_geometry_eligible_candidates": eligible,
            "ranked_candidate_parents_scored": scored,
            "ranked_agreements": agreements,
            "ranked_edges_added": added,
            "ranked_reassignments": reassignments,
            "ranked_node_or_coordinate_changes": graph_node_changes,
            "graph_gpu_groups_used": graph_gpu_groups,
            "localization_candidate_nodes": localization_candidates,
            "localization_nodes_moved": moved,
            "localization_boundary_rejected": boundary_rejected,
            "localization_rounded_coordinate_changes": rounded_changes,
            "localization_global_gate_failures": global_failures,
            "localization_node_count_changes": localization_node_changes,
            "localization_edge_changes": localization_edge_changes,
            "localization_gpu_groups_used": localization_gpu_groups,
        },
    )

    candidate_validator = aggregate_validator(validator_path)
    baseline = aggregate_validator(baseline_validator)
    candidate_errors = validator_node_errors(validator_path)
    baseline_errors = validator_node_errors(baseline_validator)
    if (
        candidate_validator["stems"] != baseline["stems"]
        or candidate_errors["by_stem"].keys() != baseline_errors["by_stem"].keys()
    ):
        raise RuntimeError("composition and public-control validator stems differ")
    for label, evidence_key, observed in (
        ("validator_adjusted_edge_jaccard", "validator_adjusted_edge_jaccard", candidate_validator["weighted_adjusted_edge_jaccard"]),
        ("validator_division_tp", "validator_division_tp", candidate_validator["division_tp"]),
        ("validator_division_fp", "validator_division_fp", candidate_validator["division_fp"]),
        ("validator_division_fn", "validator_division_fn", candidate_validator["division_fn"]),
        ("validator_division_jaccard", "validator_division_jaccard", candidate_validator["division_jaccard"]),
        ("validator_proxy_score", "validator_proxy_score", candidate_validator["proxy_score"]),
        ("validator_missed_gt_nodes", "validator_missed_gt_nodes", candidate_errors["missed_gt_nodes"]),
        ("validator_spurious_pred_nodes", "validator_spurious_pred_nodes", candidate_errors["spurious_pred_nodes"]),
    ):
        assert_close(label, evidence.get(evidence_key), float(observed))

    graph_validator = graph_promotion["candidate_validator"]
    localization_validator = localization_promotion["candidate_validator"]
    localization_errors = localization_promotion.get("candidate_node_errors", {})
    localization_by_stem = localization_errors.get("by_stem", {})
    if localization_by_stem.keys() != candidate_errors["by_stem"].keys():
        raise RuntimeError("composition and localization promotion stems differ")
    node_nonregressive_to_localization = all(
        candidate_errors["by_stem"][stem]["missed_gt_nodes"]
        <= localization_by_stem[stem]["missed_gt_nodes"]
        and candidate_errors["by_stem"][stem]["spurious_pred_nodes"]
        <= localization_by_stem[stem]["spurious_pred_nodes"]
        for stem in localization_by_stem
    )
    baseline_node_nonregressive = all(
        candidate_errors["by_stem"][stem]["missed_gt_nodes"]
        <= baseline_errors["by_stem"][stem]["missed_gt_nodes"]
        for stem in baseline_errors["by_stem"]
    )
    proxy_gain = candidate_validator["proxy_score"] - baseline["proxy_score"]
    adjusted_edge_delta = (
        candidate_validator["weighted_adjusted_edge_jaccard"]
        - baseline["weighted_adjusted_edge_jaccard"]
    )
    best_component_proxy = max(
        float(graph_validator["proxy_score"]),
        float(localization_validator["proxy_score"]),
    )
    composition_gain = candidate_validator["proxy_score"] - best_component_proxy
    missed_gain = baseline_errors["missed_gt_nodes"] - candidate_errors["missed_gt_nodes"]
    spurious_delta = (
        candidate_errors["spurious_pred_nodes"] - baseline_errors["spurious_pred_nodes"]
    )
    if not (
        proxy_gain >= MINIMUM_PROXY_GAIN
        and adjusted_edge_delta >= -MAXIMUM_ADJUSTED_EDGE_REGRESSION
        and composition_gain >= MINIMUM_COMPOSITION_GAIN_OVER_BEST_COMPONENT
        and candidate_validator["division_tp"] >= graph_validator["division_tp"]
        and candidate_validator["division_jaccard"] >= graph_validator["division_jaccard"]
        and missed_gain > 0
        and spurious_delta <= 0
        and baseline_node_nonregressive
        and node_nonregressive_to_localization
    ):
        raise RuntimeError(
            "0.945 composition promotion gate failed: "
            f"proxy_gain={proxy_gain:.6f}, composition_gain={composition_gain:.6f}, "
            f"adjusted_edge_delta={adjusted_edge_delta:.6f}, missed_gain={missed_gain}, "
            f"spurious_delta={spurious_delta}, "
            f"node_nonregressive_to_localization={node_nonregressive_to_localization}"
        )

    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": RUN_ID,
        "target_public_score": TARGET_PUBLIC_SCORE,
        "submission": submission,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": submission_sha256,
        "candidate_evidence_sha256": sha256_file(evidence_path),
        "watchdog_terminal_sha256": sha256_file(terminal_path),
        "graph_runtime_manifest_sha256": graph_runtime["manifest_sha256"],
        "localization_runtime_manifest_sha256": localization_runtime["manifest_sha256"],
        "graph_promotion_report_sha256": sha256_file(graph_promotion_report),
        "localization_promotion_report_sha256": sha256_file(localization_promotion_report),
        "graph_member_count": graph_runtime["deep_member_count"],
        "localization_member_count": localization_runtime["member_count"],
        "ranked_edges_added": added,
        "localization_nodes_moved": moved,
        "localization_rounded_coordinate_changes": rounded_changes,
        "public_control": baseline,
        "candidate_validator": candidate_validator,
        "candidate_node_errors": candidate_errors,
        "proxy_gain": proxy_gain,
        "composition_gain_over_best_component": composition_gain,
        "adjusted_edge_delta": adjusted_edge_delta,
        "missed_gt_node_gain": missed_gain,
        "spurious_pred_node_delta": spurious_delta,
        "known_public_hash_match": False,
        "competition_submission_performed": False,
        "authorized_for_submission": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-validator", type=Path, required=True)
    parser.add_argument("--graph-runtime-manifest", type=Path, required=True)
    parser.add_argument("--localization-runtime-manifest", type=Path, required=True)
    parser.add_argument("--graph-promotion-report", type=Path, required=True)
    parser.add_argument("--localization-promotion-report", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_candidate(
        args.output_root,
        args.baseline_validator,
        args.graph_runtime_manifest,
        args.localization_runtime_manifest,
        args.graph_promotion_report,
        args.localization_promotion_report,
        expected_baseline_sha256=PUBLIC_CONTROL_VALIDATOR_SHA256,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
