from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASET = runpy.run_path(str(ROOT / "scripts/build-temporal-localization-candidate-dataset.py"))
FIXTURE = runpy.run_path(str(ROOT / "tests/test_temporal_localization_candidate_dataset.py"))
VERIFIER = runpy.run_path(str(ROOT / "scripts/verify-temporal-localization-submission-candidate.py"))
verify_candidate = VERIFIER["verify_candidate"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_runtime(tmp_path: Path) -> Path:
    results = FIXTURE["result_fixture"](tmp_path / "training")
    runtime = tmp_path / "runtime"
    DATASET["stage_dataset"](results, runtime)
    return runtime


def validator_rows(*, candidate: bool) -> list[dict]:
    return [
        {
            "stem": "a",
            "weight": 100,
            "adjusted_edge_jaccard": 0.912 if candidate else 0.900,
            "div_tp": 5,
            "div_fp": 1,
            "div_fn": 1,
            "missed_gt_nodes": 8 if candidate else 10,
            "spurious_pred_nodes": 3,
        },
        {
            "stem": "b",
            "weight": 100,
            "adjusted_edge_jaccard": 0.908 if candidate else 0.900,
            "div_tp": 5,
            "div_fp": 1,
            "div_fn": 1,
            "missed_gt_nodes": 10,
            "spurious_pred_nodes": 3,
        },
    ]


def output_fixture(tmp_path: Path, runtime: Path) -> tuple[Path, Path]:
    output = tmp_path / "output"
    output.mkdir()
    submission = output / "submission.csv"
    with submission.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"))
        writer.writerow((0, "test", "node", 1, 0, 1, 2, 3, -1, -1))
        writer.writerow((1, "test", "node", 2, 1, 1, 3, 3, -1, -1))
        writer.writerow((2, "test", "edge", -1, -1, -1, -1, -1, 1, 2))
    submission_hash = hashlib.sha256(submission.read_bytes()).hexdigest()
    (output / "watchdog-terminal.json").write_text(
        json.dumps(
            {
                "run_id": "ema-temporal-localization-candidate-v1",
                "status": "completed",
                "submission_exists": True,
                "submission_sha256": submission_hash,
                "declared_budget_seconds": 39600,
                "safety_margin_seconds": 1200,
                "elapsed_seconds": 1000,
            }
        ),
        encoding="utf-8",
    )
    run_row = {
        "dataset": "test",
        "ranked_consensus_localization_candidate_nodes": 100,
        "ranked_consensus_localization_nodes_moved": 2,
        "ranked_consensus_localization_rounded_coordinate_changes": 2,
        "ranked_consensus_localization_boundary_rejected": 0,
        "ranked_consensus_localization_global_gate_failures": 0,
        "ranked_consensus_localization_node_count_changes": 0,
        "ranked_consensus_localization_edge_changes": 0,
        "ranked_consensus_localization_gpu_groups_used": 2,
    }
    write_csv(output / "run_stats.csv", [run_row])
    candidate_rows = validator_rows(candidate=True)
    write_csv(output / "validator_results.csv", candidate_rows)
    baseline = tmp_path / "baseline-validator.csv"
    write_csv(baseline, validator_rows(candidate=False))
    policy = json.loads((runtime / "temporal-localization-consensus-policy.json").read_text())
    adjusted = sum(row["adjusted_edge_jaccard"] * row["weight"] for row in candidate_rows) / sum(row["weight"] for row in candidate_rows)
    div_tp = sum(row["div_tp"] for row in candidate_rows)
    div_fp = sum(row["div_fp"] for row in candidate_rows)
    div_fn = sum(row["div_fn"] for row in candidate_rows)
    division = div_tp / (div_tp + div_fp + div_fn)
    evidence = {
        "schema_version": 1,
        "status": "completed_pending_external_promotion_gate",
        "run_id": "ema-temporal-localization-candidate-v1",
        "target_public_score": 0.945,
        "public_lineage_attributed": True,
        "public_predictions_copied": False,
        "localization_family": "temporal_convnext_axial_node_localizer_v1",
        "localization_member_count": len(policy["localization_members"]),
        "parameters_per_member": 71_249_805,
        "localization_model_sha256": [row["model_sha256"] for row in policy["localization_members"]],
        "runtime_manifest_sha256": hashlib.sha256((runtime / "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json").read_bytes()).hexdigest(),
        "node_count_preserving": True,
        "topology_preserving": True,
        "model_subset_searched_on_audit": False,
        "weights_searched_on_development": False,
        "threshold_searched_on_development": False,
        "localization_candidate_nodes": 100,
        "localization_nodes_moved": 2,
        "localization_rounded_coordinate_changes": 2,
        "localization_boundary_rejected": 0,
        "localization_global_gate_failures": 0,
        "localization_node_count_changes": 0,
        "localization_edge_changes": 0,
        "localization_gpu_groups_used": 2,
        "validator_adjusted_edge_jaccard": adjusted,
        "validator_division_tp": div_tp,
        "validator_division_fp": div_fp,
        "validator_division_fn": div_fn,
        "validator_division_jaccard": division,
        "validator_proxy_score": adjusted + 0.1 * division,
        "validator_missed_gt_nodes": sum(row["missed_gt_nodes"] for row in candidate_rows),
        "validator_spurious_pred_nodes": sum(row["spurious_pred_nodes"] for row in candidate_rows),
        "submission_sha256": submission_hash,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    (output / "candidate_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    return output, baseline


def test_candidate_is_promoted_only_after_clean_localization_gain(tmp_path: Path) -> None:
    runtime = stage_runtime(tmp_path)
    output, baseline = output_fixture(tmp_path, runtime)
    result = verify_candidate(
        output,
        baseline,
        runtime / "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json",
    )
    assert result["status"] == "eligible_for_submission"
    assert result["missed_gt_node_gain"] == 2
    assert result["localization_node_count_changes"] == 0
    assert result["localization_edge_changes"] == 0


def test_candidate_rejects_per_movie_localization_regression(tmp_path: Path) -> None:
    runtime = stage_runtime(tmp_path)
    output, baseline = output_fixture(tmp_path, runtime)
    rows = validator_rows(candidate=True)
    rows[0]["missed_gt_nodes"] = 11
    rows[1]["missed_gt_nodes"] = 7
    write_csv(output / "validator_results.csv", rows)
    evidence_path = output / "candidate_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["validator_missed_gt_nodes"] = 18
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(RuntimeError, match="promotion gate failed"):
        verify_candidate(
            output,
            baseline,
            runtime / "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json",
        )
