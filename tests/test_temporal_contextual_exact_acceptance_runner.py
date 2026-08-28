from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-temporal-contextual-exact-acceptance.py"
SPEC = importlib.util.spec_from_file_location("contextual_exact_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_exact_acceptance_command_is_pinned_and_submission_free() -> None:
    command = runner.scoring_command(
        control_csv=Path("/control.csv"),
        candidate_csv=Path("/candidate.csv"),
        truth_dir=Path("/truth"),
        scorer_lock=Path("/scorer-lock.json"),
        organizer_checkout=Path("/organizer"),
        tracksdata_checkout=Path("/tracksdata"),
        materialization_result=Path("/materialization.json"),
        output=Path("/acceptance.json"),
    )
    rendered = " ".join(command)
    assert command[1:3] == [
        "-m",
        "research.trackastra_graph.score_dual_fold_processed_candidate",
    ]
    assert "--scorer-lock" in command
    assert "--materialization-result" in command
    assert "submit" not in rendered.casefold()
    assert "leaderboard" not in rendered.casefold()


def test_exact_acceptance_evidence_requires_every_gate(tmp_path: Path) -> None:
    control = tmp_path / "control.csv"
    candidate = tmp_path / "candidate.csv"
    materialization = tmp_path / "materialization.json"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")
    materialization.write_text("{}\n", encoding="utf-8")
    checks = {
        "control_score_reproduced": True,
        "pooled_score_improved": True,
        "per_movie_regression_floor_passed": True,
        "node_recall_identical": True,
        "edge_sets_differ": True,
    }
    evidence = {
        "schema_version": 1,
        "status": "accepted",
        "run_id": runner.RUN_ID,
        "evaluation_kind": runner.EVALUATION_KIND,
        "candidate_family": runner.CANDIDATE_FAMILY,
        "appearance_family": runner.APPEARANCE_FAMILY,
        "exact_processed_gate_passed": True,
        "gate": {"checks": checks, "worst_movie_score_delta": -0.001},
        "candidate_node_rows_identical": True,
        "candidate_edge_sets_differ": True,
        "processed_control_sha256": runner.sha256_file(control),
        "processed_candidate_sha256": runner.sha256_file(candidate),
        "materialization_result_sha256": runner.sha256_file(materialization),
        "authorized_for_submission": False,
        "competition_submission_performed": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
    }
    runner.checked_exact_acceptance(
        evidence,
        control_csv=control,
        candidate_csv=candidate,
        materialization_result=materialization,
    )
    evidence["gate"]["checks"]["pooled_score_improved"] = False
    with pytest.raises(RuntimeError, match="exact CPU acceptance"):
        runner.checked_exact_acceptance(
            evidence,
            control_csv=control,
            candidate_csv=candidate,
            materialization_result=materialization,
        )


def test_processed_launcher_chain_is_hash_bound(tmp_path: Path) -> None:
    candidate = tmp_path / "processed_candidate.csv"
    materialization = tmp_path / "materialization_result.json"
    candidate.write_bytes(b"candidate")
    materialization.write_text("{}\n", encoding="utf-8")
    launcher = {
        "schema_version": 1,
        "status": "completed",
        "run_id": "temporal-contextual-pair-fusion-processed-acceptance-v3",
        "gpu_count": 2,
        "processed_control_sha256": runner.EXPECTED_PROCESSED_CONTROL_SHA256,
        "materialization_result_sha256": runner.sha256_file(materialization),
        "processed_candidate_sha256": runner.sha256_file(candidate),
        "total_changed_edges": 3,
        "processed_ground_truth_read": False,
        "exact_processed_scoring_performed": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "authorized_for_exact_cpu_scoring": True,
        "authorized_for_submission": False,
    }
    runner.checked_cloud_processed_launcher(launcher, materialization, candidate)
    candidate.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="launcher evidence"):
        runner.checked_cloud_processed_launcher(
            launcher, materialization, candidate
        )


def test_exact_acceptance_script_has_no_submission_command() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "kaggle competitions" not in source
    assert "competitions submit" not in source
    assert "public_leaderboard_used_for_selection" in source
