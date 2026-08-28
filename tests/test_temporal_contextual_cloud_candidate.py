from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-temporal-contextual-cloud-candidate.py"
SPEC = importlib.util.spec_from_file_location("contextual_cloud_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)


def test_cloud_candidate_command_is_frozen_and_submission_free() -> None:
    command = cloud.candidate_command(
        runtime_root=Path("/runtime"),
        base_submission=Path("/base.csv"),
        base_graph_root=Path("/raw"),
        image_root=Path("/test"),
        trackastra_root=Path("/trackastra"),
        appearance_root=Path("/appearance"),
        acceptance_evidence=Path("/acceptance.json"),
        output_dir=Path("/output"),
    )
    rendered = " ".join(command)
    assert "dual_fold_appearance_submission.py" in rendered
    assert "--orchestrate" in command
    assert command[command.index("--max-tokens") + 1] == "512"
    assert command[command.index("--candidate-radius") + 1] == "80.0"
    assert command[command.index("--node-batch-size") + 1] == "64"
    assert command[command.index("--hard-stop-seconds") + 1] == "36000"
    assert "submit" not in rendered.casefold()
    assert "leaderboard" not in rendered.casefold()


def exact_acceptance() -> dict:
    return {
        "schema_version": 1,
        "status": "accepted",
        "run_id": cloud.EXACT_RUN_ID,
        "evaluation_kind": "exact_processed_dual_fold_acceptance",
        "exact_processed_gate_passed": True,
        "candidate_family": cloud.CANDIDATE_FAMILY,
        "appearance_family": cloud.APPEARANCE_FAMILY,
        "gate": {
            "checks": {
                "control_score_reproduced": True,
                "pooled_score_improved": True,
                "per_movie_regression_floor_passed": True,
                "node_recall_identical": True,
                "edge_sets_differ": True,
            }
        },
        "models": {fold: {} for fold in cloud.FOLDS},
        "appearance_models": {fold: {} for fold in cloud.FOLDS},
        "appearance_blend": {fold: {} for fold in cloud.FOLDS},
        "candidate_node_rows_identical": True,
        "candidate_edge_sets_differ": True,
        "authorized_for_submission": False,
        "competition_submission_performed": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
    }


def test_cloud_candidate_acceptance_check_fails_closed() -> None:
    evidence = exact_acceptance()
    cloud.checked_exact_acceptance(evidence)
    evidence["gate"]["checks"]["node_recall_identical"] = False
    with pytest.raises(RuntimeError, match="exact acceptance"):
        cloud.checked_exact_acceptance(evidence)


def test_cloud_candidate_report_is_hash_bound(tmp_path: Path) -> None:
    candidate = tmp_path / "submission.csv"
    acceptance = tmp_path / "acceptance.json"
    candidate.write_bytes(b"candidate")
    acceptance.write_bytes(b"acceptance")
    report = {
        "schema_version": 1,
        "status": "completed",
        "candidate_family": cloud.CANDIDATE_FAMILY,
        "appearance_family": cloud.APPEARANCE_FAMILY,
        "gpu_count": 2,
        "inference_hard_stop_seconds": cloud.INFERENCE_HARD_STOP_SECONDS,
        "notebook_runtime_reserve_seconds": cloud.FINALIZATION_RESERVE_SECONDS,
        "whole_movie_coverage": ["44b6_a", "6bba_b"],
        "base_submission_sha256": cloud.EXPECTED_BASE_SHA256,
        "candidate_submission_sha256": cloud.sha256_file(candidate),
        "acceptance_evidence_sha256": cloud.sha256_file(acceptance),
        "models": {fold: {} for fold in cloud.FOLDS},
        "appearance_models": {fold: {} for fold in cloud.FOLDS},
        "appearance_blend": {fold: {} for fold in cloud.FOLDS},
        "total_changed_edges": 9,
        "nodes_preserved_exactly": True,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }
    cloud.checked_candidate(
        report, candidate_csv=candidate, acceptance_evidence=acceptance
    )
    report["candidate_submission_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="final candidate"):
        cloud.checked_candidate(
            report, candidate_csv=candidate, acceptance_evidence=acceptance
        )


def test_cloud_candidate_json_parser_rejects_non_object() -> None:
    completed = subprocess.CompletedProcess(["fake"], 0, stdout="[]", stderr="")
    with pytest.raises(RuntimeError, match="JSON object"):
        cloud.parse_json_output(completed, "fake verifier")


def test_cloud_candidate_script_has_no_submission_command() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "kaggle competitions" not in source
    assert "competitions submit" not in source
    assert '"ready_for_submission_upload": true' in source
