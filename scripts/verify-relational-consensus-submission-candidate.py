#!/usr/bin/env python
"""Verify and promote an additive relational-consensus Kaggle output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_COMMON = runpy.run_path(str(ROOT / "scripts/verify-learned-division-submission-candidate.py"))
_RANKED = runpy.run_path(str(ROOT / "scripts/verify-ranked-consensus-submission-candidate.py"))
aggregate_validator = _COMMON["aggregate_validator"]
read_csv_rows = _COMMON["read_csv_rows"]
sha256_file = _COMMON["sha256_file"]
unique_file = _COMMON["unique_file"]
validate_submission_csv = _COMMON["validate_submission_csv"]
_assert_close = _COMMON["_assert_close"]
_sum_integral = _RANKED["_sum_integral"]

RUN_ID = "ema-relational-consensus-candidate-v1"
RUNTIME_RUN_ID = "competition-relational-consensus-division-dataset-v1"
POLICY_RUN_ID = "competition-relational-consensus-division-policy-v1"
EXPECTED_PARAMETER_COUNT = 48_313_050
KNOWN_PUBLIC_SUBMISSION_SHA256 = _COMMON["KNOWN_PUBLIC_SUBMISSION_SHA256"]
MINIMUM_PROXY_GAIN = 0.005
MAXIMUM_ADJUSTED_EDGE_REGRESSION = 0.001
PUBLIC_CONTROL_VALIDATOR_SHA256 = _COMMON["PUBLIC_CONTROL_VALIDATOR_SHA256"]


def validate_runtime(runtime_manifest: Path) -> dict[str, Any]:
    root = runtime_manifest.parent
    manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    policy_path = root / "relational-consensus-policy.json"
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUNTIME_RUN_ID
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and "relational-consensus-policy.json" in files
        and policy_path.is_file()
    ):
        raise RuntimeError("relational runtime manifest is invalid")
    for name, record in files.items():
        path = root / name
        if not (
            path.is_file()
            and record.get("path") == name
            and record.get("sha256") == sha256_file(path)
        ):
            raise RuntimeError(f"relational runtime file changed: {name}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    members = policy.get("relational_members", [])
    hashes = [row.get("model_sha256") for row in members]
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "development_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("relational_policy")
        in {"equal_rank_selection_admitted_ensemble", "strongest_selection_individual"}
        and 1 <= len(members) <= 8
        and policy.get("relational_member_count") == len(members)
        and len(set(hashes)) == len(hashes)
        and all(
            row.get("path") in files
            and row.get("model_sha256") == sha256_file(root / row["path"])
            and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("selection_gate_passed") is True
            and row.get("audit_gate_passed") is True
            for row in members
        )
        and policy.get("base_safe_division_heuristic_enabled") is True
        and policy.get("external_policy_additive_only") is True
        and policy.get("maximum_added_edges_per_movie") == 1
        and policy.get("exact_two_t4_required") is True
        and policy.get("absolute_threshold_used") is False
        and policy.get("model_subset_searched_on_audit") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
    ):
        raise RuntimeError("relational runtime policy is invalid")
    return {
        "manifest_sha256": sha256_file(runtime_manifest),
        "policy_sha256": sha256_file(policy_path),
        "deep_policy": policy["relational_policy"],
        "deep_member_count": len(members),
        "deep_model_sha256": hashes,
        "morphology_model_sha256": policy["morphology_model_sha256"],
        "expected_gpu_groups": 2 if len(members) >= 2 else 1,
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
        and float(terminal.get("elapsed_seconds", math.inf)) < 10_800.0
    ):
        raise RuntimeError("relational candidate watchdog terminal is invalid")
    if not (
        evidence.get("schema_version") == 1
        and evidence.get("status") == "completed_pending_external_promotion_gate"
        and evidence.get("run_id") == RUN_ID
        and float(evidence.get("target_public_score", 0.0)) == 0.945
        and evidence.get("public_lineage_attributed") is True
        and evidence.get("public_predictions_copied") is False
        and evidence.get("deep_policy") == runtime["deep_policy"]
        and evidence.get("deep_member_count") == runtime["deep_member_count"]
        and evidence.get("deep_parameter_count_per_member") == EXPECTED_PARAMETER_COUNT
        and evidence.get("deep_model_sha256") == runtime["deep_model_sha256"]
        and evidence.get("morphology_model_sha256") == runtime["morphology_model_sha256"]
        and evidence.get("absolute_threshold_used") is False
        and evidence.get("runtime_manifest_sha256") == runtime["manifest_sha256"]
        and evidence.get("base_safe_division_heuristic_enabled") is True
        and evidence.get("external_policy_additive_only") is True
        and evidence.get("competition_submission_performed") is False
        and evidence.get("authorized_for_submission") is False
        and evidence.get("submission_sha256") == submission_sha256
    ):
        raise RuntimeError("relational candidate evidence is invalid")
    if submission_sha256 in KNOWN_PUBLIC_SUBMISSION_SHA256:
        raise RuntimeError("relational candidate is identical to an audited public output")
    submission = validate_submission_csv(submission_path)
    run_rows = read_csv_rows(run_stats_path)
    if not run_rows:
        raise RuntimeError("relational candidate run stats are empty")
    geometric = _sum_integral(run_rows, "ranked_consensus_geometric_candidates")
    eligible = _sum_integral(run_rows, "ranked_consensus_geometry_eligible_candidates")
    scored = _sum_integral(run_rows, "ranked_consensus_candidate_parents_scored")
    agreements = _sum_integral(run_rows, "ranked_consensus_ranking_agreed")
    added = _sum_integral(run_rows, "ranked_consensus_added_edges")
    maximum_additions = _sum_integral(run_rows, "ranked_consensus_maximum_additions")
    absolute_threshold = _sum_integral(run_rows, "ranked_consensus_absolute_threshold_used")
    reassignments = _sum_integral(run_rows, "ranked_consensus_reassignment_performed")
    node_changes = _sum_integral(run_rows, "ranked_consensus_node_or_coordinate_changes")
    gpu_groups = _sum_integral(run_rows, "ranked_consensus_gpu_groups_used")
    if not (
        geometric > 0
        and 0 < eligible <= geometric
        and scored == eligible
        and 0 < agreements == added <= maximum_additions
        and absolute_threshold == 0
        and reassignments == 0
        and node_changes == 0
        and gpu_groups == len(run_rows) * runtime["expected_gpu_groups"]
    ):
        raise RuntimeError("relational consensus integrity gate failed")
    for label, observed, expected in (
        ("ranked_geometric_candidates", evidence.get("ranked_geometric_candidates"), geometric),
        ("ranked_geometry_eligible_candidates", evidence.get("ranked_geometry_eligible_candidates"), eligible),
        ("ranked_candidate_parents_scored", evidence.get("ranked_candidate_parents_scored"), scored),
        ("ranked_agreements", evidence.get("ranked_agreements"), agreements),
        ("ranked_edges_added", evidence.get("ranked_edges_added"), added),
        ("ranked_reassignments", evidence.get("ranked_reassignments"), reassignments),
        ("ranked_node_or_coordinate_changes", evidence.get("ranked_node_or_coordinate_changes"), node_changes),
    ):
        _assert_close(label, observed, float(expected))
    candidate_validator = aggregate_validator(validator_path)
    baseline = aggregate_validator(baseline_validator)
    if candidate_validator["stems"] != baseline["stems"]:
        raise RuntimeError("relational candidate and public-control validator stems differ")
    for label, evidence_key, aggregate_key in (
        ("validator_adjusted_edge_jaccard", "validator_adjusted_edge_jaccard", "weighted_adjusted_edge_jaccard"),
        ("validator_division_tp", "validator_division_tp", "division_tp"),
        ("validator_division_fp", "validator_division_fp", "division_fp"),
        ("validator_division_fn", "validator_division_fn", "division_fn"),
        ("validator_division_jaccard", "validator_division_jaccard", "division_jaccard"),
        ("validator_proxy_score", "validator_proxy_score", "proxy_score"),
    ):
        _assert_close(label, evidence.get(evidence_key), candidate_validator[aggregate_key])
    proxy_gain = candidate_validator["proxy_score"] - baseline["proxy_score"]
    adjusted_edge_delta = candidate_validator["weighted_adjusted_edge_jaccard"] - baseline["weighted_adjusted_edge_jaccard"]
    if not (
        proxy_gain >= MINIMUM_PROXY_GAIN
        and adjusted_edge_delta >= -MAXIMUM_ADJUSTED_EDGE_REGRESSION
        and candidate_validator["division_tp"] > baseline["division_tp"]
        and candidate_validator["division_jaccard"] > baseline["division_jaccard"]
    ):
        raise RuntimeError(
            "0.945 relational promotion gate failed: "
            f"proxy_gain={proxy_gain:.6f}, adjusted_edge_delta={adjusted_edge_delta:.6f}, "
            f"division_tp={candidate_validator['division_tp']}, division_jaccard={candidate_validator['division_jaccard']:.6f}"
        )
    return {
        "schema_version": 1,
        "status": "eligible_for_submission",
        "run_id": RUN_ID,
        "target_public_score": 0.945,
        "submission": submission,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": submission_sha256,
        "candidate_evidence_sha256": sha256_file(evidence_path),
        "watchdog_terminal_sha256": sha256_file(terminal_path),
        "runtime_manifest_sha256": runtime["manifest_sha256"],
        "deep_policy": runtime["deep_policy"],
        "deep_member_count": runtime["deep_member_count"],
        "gpu_groups_used_per_movie": runtime["expected_gpu_groups"],
        "ranked_edges_added": added,
        "ranked_agreements": agreements,
        "ranked_reassignments": reassignments,
        "ranked_node_or_coordinate_changes": node_changes,
        "base_safe_division_heuristic_enabled": True,
        "external_policy_additive_only": True,
        "public_control": baseline,
        "public_control_validator_sha256": baseline_sha256,
        "candidate_validator": candidate_validator,
        "proxy_gain": proxy_gain,
        "adjusted_edge_delta": adjusted_edge_delta,
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
