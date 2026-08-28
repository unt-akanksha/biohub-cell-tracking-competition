from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-temporal-contextual-cloud-calibration.py"
SPEC = importlib.util.spec_from_file_location("contextual_cloud_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)


def test_cloud_calibration_command_is_frozen_and_submission_free() -> None:
    command = cloud.calibration_command(
        runtime_root=Path("/runtime"),
        appearance_root=Path("/appearance"),
        trackastra_root=Path("/trackastra"),
        competition_dir=Path("/competition"),
        output_dir=Path("/output"),
    )
    rendered = " ".join(command)
    assert "calibrate_dual_fold_blend.py" in rendered
    assert "--orchestrate" in command
    assert command[command.index("--max-tokens") + 1] == "512"
    assert command[command.index("--candidate-radius") + 1] == "80.0"
    assert command[command.index("--node-batch-size") + 1] == "64"
    assert command[command.index("--max-wall-seconds") + 1] == "18000"
    assert command[command.index("--orchestrator-hard-stop-seconds") + 1] == "19800"
    assert "submit" not in rendered.casefold()
    assert "leaderboard" not in rendered.casefold()


def calibration_fixture() -> dict:
    zero_control = {
        "ensemble_mode": "target_only",
        "appearance_weight": 0.0,
        "division_weight": 0.0,
    }
    selected = {
        "eligible": True,
        "pooled_gain_vs_zero": 0.0015,
        "worst_movie_delta_vs_zero": -0.001,
    }
    folds = {
        fold: {
            "schema_version": 1,
            "status": "completed",
            "run_id": cloud.RUN_ID,
            "appearance_family": cloud.APPEARANCE_FAMILY,
            "calibration_stems": [f"movie-{index}" for index in range(12)],
            "selection": {
                "improved": True,
                "selected": dict(selected),
                "grid": [dict(zero_control)],
            },
            "processed_acceptance_ground_truth_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        for fold in cloud.FOLDS
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "run_id": cloud.RUN_ID,
        "appearance_family": cloud.APPEARANCE_FAMILY,
        "gpu_count": 2,
        "both_folds_improved": True,
        "folds": folds,
        "processed_acceptance_ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }


def test_cloud_calibration_evidence_checks_fail_closed(tmp_path: Path) -> None:
    appearance_root = tmp_path / "appearance"
    appearance_root.mkdir()
    training_terminal = appearance_root / "training_terminal.json"
    training_terminal.write_text("{}\n", encoding="utf-8")
    transfer = {
        "schema_version": 1,
        "status": "completed",
        "run_id": cloud.TRANSFER_RUN_ID,
        "gpu_count": 2,
        "runtime_manifest_sha256": cloud.EXPECTED_RUNTIME_MANIFEST_SHA256,
        "source_hashes": {
            "temporal_source_tree_sha256": (
                cloud.EXPECTED_TEMPORAL_SOURCE_TREE_SHA256
            )
        },
        "training_terminal_sha256": cloud.sha256_file(training_terminal),
        "strict_checkpoint_loaded": True,
        "authorized_for_calibration": True,
        "authorized_for_submission": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    cloud.checked_transfer_launcher(transfer, appearance_root)
    transfer["authorized_for_submission"] = True
    with pytest.raises(RuntimeError, match="launcher"):
        cloud.checked_transfer_launcher(transfer, appearance_root)

    calibration = calibration_fixture()
    cloud.checked_calibration(calibration)
    calibration["folds"]["target_44b6"]["selection"]["selected"][
        "pooled_gain_vs_zero"
    ] = 0.0009
    with pytest.raises(RuntimeError, match="calibration output"):
        cloud.checked_calibration(calibration)


def test_cloud_calibration_json_parser_rejects_non_object() -> None:
    completed = subprocess.CompletedProcess(["fake"], 0, stdout="[]", stderr="")
    with pytest.raises(RuntimeError, match="JSON object"):
        cloud.parse_json_output(completed, "fake verifier")


def test_cloud_calibration_script_has_no_submission_command() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "kaggle competitions" not in source
    assert "competitions submit" not in source
    assert '"submission_created": false' in source
