#!/usr/bin/env python
"""Verify and promote an EMA plus temporal-localization Kaggle output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(str(ROOT / "scripts/verify-learned-division-submission-candidate.py"))
RANKED = runpy.run_path(str(ROOT / "scripts/verify-ranked-consensus-submission-candidate.py"))
aggregate_validator = COMMON["aggregate_validator"]
read_csv_rows = COMMON["read_csv_rows"]
sha256_file = COMMON["sha256_file"]
unique_file = COMMON["unique_file"]
validate_submission_csv = COMMON["validate_submission_csv"]
assert_close = COMMON["_assert_close"]
sum_integral = RANKED["_sum_integral"]

RUN_ID = "ema-temporal-localization-candidate-v1"
RUNTIME_RUN_ID = "competition-temporal-localization-consensus-dataset-v1"
POLICY_RUN_ID = "competition-temporal-localization-consensus-policy-v1"
EXPECTED_PARAMETER_COUNT = 71_249_805
KNOWN_PUBLIC_SUBMISSION_SHA256 = COMMON["KNOWN_PUBLIC_SUBMISSION_SHA256"]
PUBLIC_CONTROL_VALIDATOR_SHA256 = COMMON["PUBLIC_CONTROL_VALIDATOR_SHA256"]
MINIMUM_PROXY_GAIN = 0.005
MAXIMUM_ADJUSTED_EDGE_REGRESSION = 0.001


def validate_runtime(runtime_manifest: Path) -> dict[str, Any]:
    root = runtime_manifest.parent
    manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    policy_path = root / "temporal-localization-consensus-policy.json"
    development_path = root / "evidence/real-development-probe.json"
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUNTIME_RUN_ID
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and policy_path.is_file()
        and development_path.is_file()
        and "temporal-localization-consensus-policy.json" in files
        and "evidence/real-development-probe.json" in files
    ):
        raise RuntimeError("temporal localization runtime manifest is invalid")
    for name, record in files.items():
        path = root / name
        if not (
            path.is_file()
            and record.get("path") == name
            and record.get("sha256") == sha256_file(path)
        ):
            raise RuntimeError(f"temporal localization runtime file changed: {name}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    members = policy.get("localization_members", [])
    hashes = [row.get("model_sha256") for row in members]
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "development_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and 3 <= len(members) <= 4
        and policy.get("localization_member_count") == len(members)
        and len(set(hashes)) == len(members)
        and all(
            row.get("path") in files
            and row.get("terminal_path") in files
            and row.get("model_sha256") == sha256_file(root / row["path"])
            and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("selection_gate_passed") is True
            and row.get("audit_gate_passed") is True
            and row.get("division_critical_selection_gate_passed") is True
            and row.get("division_critical_audit_gate_passed") is True
            and row.get("real_selection_gate_passed") is True
            and row.get("real_audit_gate_passed") is True
            and row.get("real_division_critical_selection_gate_passed") is True
            and row.get("real_division_critical_audit_gate_passed") is True
            and row.get("serialized_checkpoint_selection_gate_passed") is True
            for row in members
        )
        and policy.get("ensemble_policy") == "equal_mean_all_dual_domain_eligible_members"
        and policy.get("real_replay_probability") == 0.25
        and policy.get("minimum_members") == 3
        and policy.get("blend") == 0.75
        and policy.get("minimum_correction_um") == 2.0
        and policy.get("maximum_correction_um") == 9.5
        and policy.get("maximum_member_disagreement_um") == 1.5
        and policy.get("maximum_predicted_sigma_um") == 2.5
        and policy.get("maximum_safe_probability") == 0.35
        and policy.get("minimum_direction_cosine") == 0.8
        and policy.get("maximum_move_fraction") == 0.1
        and policy.get("minimum_forced_division_critical_fraction") == 0.25
        and policy.get("division_critical_selection_gate_required") is True
        and policy.get("division_critical_audit_gate_required") is True
        and policy.get("real_selection_gate_required") is True
        and policy.get("real_audit_gate_required") is True
        and policy.get("real_division_critical_selection_gate_required") is True
        and policy.get("real_division_critical_audit_gate_required") is True
        and policy.get("node_count_preserving") is True
        and policy.get("topology_preserving") is True
        and policy.get("exact_two_t4_required") is True
        and policy.get("model_subset_searched_on_audit") is False
        and policy.get("weights_searched_on_development") is False
        and policy.get("threshold_searched_on_development") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
        and development.get("status") == "development_passed"
        and development.get("gate", {}).get("passed") is True
        and development.get("policy_or_member_selection_performed") is False
        and development.get("authorized_for_submission") is False
    ):
        raise RuntimeError("temporal localization runtime policy is invalid")
    return {
        "manifest_sha256": sha256_file(runtime_manifest),
        "policy_sha256": sha256_file(policy_path),
        "member_count": len(members),
        "model_sha256": hashes,
        "expected_gpu_groups": 2,
    }


def validator_node_errors(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    required = {"stem", "missed_gt_nodes", "spurious_pred_nodes"}
    if not rows or not required <= set(rows[0]):
        raise RuntimeError("validator node-error columns changed")
    by_stem = {
        row["stem"]: {
            "missed_gt_nodes": int(float(row["missed_gt_nodes"])),
            "spurious_pred_nodes": int(float(row["spurious_pred_nodes"])),
        }
        for row in rows
    }
    if len(by_stem) != len(rows):
        raise RuntimeError("validator contains duplicate node-error stems")
    return {
        "by_stem": by_stem,
        "missed_gt_nodes": sum(row["missed_gt_nodes"] for row in by_stem.values()),
        "spurious_pred_nodes": sum(row["spurious_pred_nodes"] for row in by_stem.values()),
    }


def verify_candidate(
    output_root: Path,
    baseline_validator: Path,
    runtime_manifest: Path,
    *,
    expected_baseline_sha256: str | None = None,
) -> dict[str, Any]:
    runtime = validate_runtime(runtime_manifest)
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
        raise RuntimeError("temporal localization watchdog terminal is invalid")
    if not (
        evidence.get("schema_version") == 1
        and evidence.get("status") == "completed_pending_external_promotion_gate"
        and evidence.get("run_id") == RUN_ID
        and evidence.get("target_public_score") == 0.945
        and evidence.get("public_lineage_attributed") is True
        and evidence.get("public_predictions_copied") is False
        and evidence.get("localization_family") == "temporal_convnext_axial_node_localizer_v1"
        and evidence.get("localization_member_count") == runtime["member_count"]
        and evidence.get("parameters_per_member") == EXPECTED_PARAMETER_COUNT
        and evidence.get("localization_model_sha256") == runtime["model_sha256"]
        and evidence.get("runtime_manifest_sha256") == runtime["manifest_sha256"]
        and evidence.get("node_count_preserving") is True
        and evidence.get("topology_preserving") is True
        and evidence.get("model_subset_searched_on_audit") is False
        and evidence.get("weights_searched_on_development") is False
        and evidence.get("threshold_searched_on_development") is False
        and evidence.get("competition_submission_performed") is False
        and evidence.get("authorized_for_submission") is False
        and evidence.get("submission_sha256") == submission_sha256
    ):
        raise RuntimeError("temporal localization candidate evidence is invalid")
    if submission_sha256 in KNOWN_PUBLIC_SUBMISSION_SHA256:
        raise RuntimeError("temporal localization candidate is identical to an audited public output")
    submission = validate_submission_csv(submission_path)
    run_rows = read_csv_rows(run_stats_path)
    if not run_rows:
        raise RuntimeError("temporal localization run stats are empty")
    candidate_nodes = sum_integral(run_rows, "ranked_consensus_localization_candidate_nodes")
    moved = sum_integral(run_rows, "ranked_consensus_localization_nodes_moved")
    rounded_changes = sum_integral(run_rows, "ranked_consensus_localization_rounded_coordinate_changes")
    boundary_rejected = sum_integral(run_rows, "ranked_consensus_localization_boundary_rejected")
    global_failures = sum_integral(run_rows, "ranked_consensus_localization_global_gate_failures")
    node_count_changes = sum_integral(run_rows, "ranked_consensus_localization_node_count_changes")
    edge_changes = sum_integral(run_rows, "ranked_consensus_localization_edge_changes")
    gpu_groups = sum_integral(run_rows, "ranked_consensus_localization_gpu_groups_used")
    if not (
        candidate_nodes > 0
        and 0 < rounded_changes <= moved <= candidate_nodes
        and 0 <= boundary_rejected <= candidate_nodes
        and global_failures == 0
        and node_count_changes == 0
        and edge_changes == 0
        and gpu_groups == len(run_rows) * runtime["expected_gpu_groups"]
    ):
        raise RuntimeError("temporal localization coordinate-only integrity gate failed")
    for label, key, expected in (
        ("localization_candidate_nodes", "localization_candidate_nodes", candidate_nodes),
        ("localization_nodes_moved", "localization_nodes_moved", moved),
        ("localization_rounded_coordinate_changes", "localization_rounded_coordinate_changes", rounded_changes),
        ("localization_boundary_rejected", "localization_boundary_rejected", boundary_rejected),
        ("localization_global_gate_failures", "localization_global_gate_failures", global_failures),
        ("localization_node_count_changes", "localization_node_count_changes", node_count_changes),
        ("localization_edge_changes", "localization_edge_changes", edge_changes),
        ("localization_gpu_groups_used", "localization_gpu_groups_used", gpu_groups),
    ):
        assert_close(label, evidence.get(key), float(expected))

    candidate_validator = aggregate_validator(validator_path)
    baseline = aggregate_validator(baseline_validator)
    candidate_errors = validator_node_errors(validator_path)
    baseline_errors = validator_node_errors(baseline_validator)
    if candidate_validator["stems"] != baseline["stems"] or candidate_errors["by_stem"].keys() != baseline_errors["by_stem"].keys():
        raise RuntimeError("candidate and public-control validator stems differ")
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
    per_stem_nonregressive = all(
        candidate_errors["by_stem"][stem]["missed_gt_nodes"]
        <= baseline_errors["by_stem"][stem]["missed_gt_nodes"]
        for stem in baseline_errors["by_stem"]
    )
    proxy_gain = candidate_validator["proxy_score"] - baseline["proxy_score"]
    adjusted_edge_delta = candidate_validator["weighted_adjusted_edge_jaccard"] - baseline["weighted_adjusted_edge_jaccard"]
    missed_gain = baseline_errors["missed_gt_nodes"] - candidate_errors["missed_gt_nodes"]
    spurious_delta = candidate_errors["spurious_pred_nodes"] - baseline_errors["spurious_pred_nodes"]
    if not (
        proxy_gain >= MINIMUM_PROXY_GAIN
        and adjusted_edge_delta >= -MAXIMUM_ADJUSTED_EDGE_REGRESSION
        and missed_gain > 0
        and spurious_delta <= 0
        and per_stem_nonregressive
        and candidate_validator["division_jaccard"] >= baseline["division_jaccard"]
    ):
        raise RuntimeError(
            "0.945 temporal localization promotion gate failed: "
            f"proxy_gain={proxy_gain:.6f}, adjusted_edge_delta={adjusted_edge_delta:.6f}, "
            f"missed_gain={missed_gain}, spurious_delta={spurious_delta}, "
            f"per_stem_nonregressive={per_stem_nonregressive}"
        )
    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": RUN_ID,
        "target_public_score": 0.945,
        "submission": submission,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": submission_sha256,
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "localization_member_count": runtime["member_count"],
        "localization_nodes_moved": moved,
        "localization_rounded_coordinate_changes": rounded_changes,
        "localization_node_count_changes": node_count_changes,
        "localization_edge_changes": edge_changes,
        "public_control": baseline,
        "candidate_validator": candidate_validator,
        "public_control_node_errors": baseline_errors,
        "candidate_node_errors": candidate_errors,
        "proxy_gain": proxy_gain,
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
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_candidate(
        args.output_root,
        args.baseline_validator,
        args.runtime_manifest,
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
