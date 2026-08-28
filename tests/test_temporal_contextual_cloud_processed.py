from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-temporal-contextual-cloud-processed.py"
SPEC = importlib.util.spec_from_file_location("contextual_cloud_processed", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)


def test_cloud_processed_command_is_frozen_and_submission_free() -> None:
    command = cloud.materialization_command(
        runtime_root=Path("/runtime"),
        processed_control_csv=Path("/control.csv"),
        raw_graph_root=Path("/raw"),
        competition_dir=Path("/competition"),
        trackastra_root=Path("/trackastra"),
        appearance_root=Path("/appearance"),
        calibration_terminal=Path("/calibration/calibration_terminal.json"),
        output_dir=Path("/output"),
    )
    rendered = " ".join(command)
    assert "dual_fold_appearance_processed_acceptance.py" in rendered
    assert "--orchestrate" in command
    assert command[command.index("--max-tokens") + 1] == "512"
    assert command[command.index("--candidate-radius") + 1] == "80.0"
    assert command[command.index("--node-batch-size") + 1] == "64"
    assert command[command.index("--hard-stop-seconds") + 1] == "19800"
    assert "submit" not in rendered.casefold()
    assert "leaderboard" not in rendered.casefold()


def test_artifact_tree_hash_is_path_and_content_bound(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a").write_bytes(b"one")
    first = cloud.artifact_tree_sha256(root)
    (root / "a").write_bytes(b"two")
    assert cloud.artifact_tree_sha256(root) != first


def test_processed_evidence_checks_fail_closed(tmp_path: Path) -> None:
    calibration_terminal = tmp_path / "calibration_terminal.json"
    calibration_terminal.write_text("{}\n", encoding="utf-8")
    candidate = tmp_path / "processed_candidate.csv"
    candidate.write_bytes(b"candidate")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "run_id": cloud.RUN_ID,
        "candidate_family": cloud.CANDIDATE_FAMILY,
        "appearance_family": cloud.APPEARANCE_FAMILY,
        "evaluation_kind": "predeclared_processed_candidate_materialization",
        "gpu_count": 2,
        "whole_movie_sharding": True,
        "processed_control_sha256": cloud.EXPECTED_PROCESSED_CONTROL_SHA256,
        "processed_candidate_sha256": cloud.sha256_file(candidate),
        "calibration_terminal_sha256": cloud.sha256_file(calibration_terminal),
        "datasets": {stem: {} for stem in cloud.EXPECTED_PROCESSED_STEMS},
        "appearance_models": {fold: {} for fold in cloud.FOLDS},
        "total_changed_edges": 7,
        "ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
        "exact_processed_scoring_performed": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    cloud.checked_materialization(payload, candidate, calibration_terminal)
    payload["ground_truth_read"] = True
    with pytest.raises(RuntimeError, match="materialization"):
        cloud.checked_materialization(payload, candidate, calibration_terminal)


def test_cloud_processed_json_parser_rejects_non_object() -> None:
    completed = subprocess.CompletedProcess(["fake"], 0, stdout="[]", stderr="")
    with pytest.raises(RuntimeError, match="JSON object"):
        cloud.parse_json_output(completed, "fake verifier")


def test_cloud_processed_script_has_no_submission_command() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "kaggle competitions" not in source
    assert "competitions submit" not in source
    assert '"competition_submission_performed": false' in source
