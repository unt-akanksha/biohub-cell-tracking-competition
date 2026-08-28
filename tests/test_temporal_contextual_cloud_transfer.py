from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-temporal-contextual-cloud-transfer.py"
SPEC = importlib.util.spec_from_file_location("contextual_cloud_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)


def test_cloud_transfer_binds_repository_sources_and_frozen_recipe() -> None:
    observed = cloud.validate_repository_sources(ROOT)
    assert {
        key: value for key, value in observed.items() if key in cloud.SOURCE_HASHES
    } == cloud.SOURCE_HASHES
    assert observed["temporal_source_file_count"] == "26"
    assert observed["temporal_source_tree_sha256"] == (
        cloud.EXPECTED_TEMPORAL_SOURCE_TREE_SHA256
    )

    command = cloud.transfer_command(
        runtime_root=Path("/runtime"),
        competition_dir=Path("/competition"),
        synthetic_root=Path("/synthetic"),
        pretraining_root=Path("/pretraining"),
        output_dir=Path("/output"),
    )
    rendered = " ".join(command)
    assert "train_dual_fold_contextual_pair_fusion.py" in rendered
    assert "--orchestrate" in command
    assert command[command.index("--steps") + 1] == "20000"
    assert command[command.index("--real-replay-probability") + 1] == "0.60"
    assert command[command.index("--minimum-real-composite-gain") + 1] == "0.005"
    assert command[command.index("--maximum-synthetic-metric-regression") + 1] == "0.01"
    assert command[command.index("--max-wall-seconds") + 1] == "36000"
    assert command[command.index("--orchestrator-hard-stop-seconds") + 1] == "37800"
    assert "submit" not in rendered.casefold()
    assert "leaderboard" not in rendered.casefold()


def test_cloud_evidence_checks_fail_closed() -> None:
    acceptance = {
        "schema_version": 1,
        "status": "accepted_verified",
        "run_id": "zebrahub-contextual-acceptance-evaluation-v1",
        "gpu_count": 2,
        "both_folds_improved": True,
        "folds": {"target_44b6": {}, "target_6bba": {}},
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    cloud.checked_acceptance(acceptance)
    acceptance["public_leaderboard_used_for_selection"] = True
    with pytest.raises(RuntimeError, match="prerequisite"):
        cloud.checked_acceptance(acceptance)

    training = {
        "schema_version": 1,
        "status": "verified",
        "run_id": cloud.RUN_ID,
        "appearance_family": cloud.APPEARANCE_FAMILY,
        "gpu_count": 2,
        "strict_checkpoint_loaded": True,
        "competition_artifacts_found": False,
        "authorized_for_calibration": True,
        "authorized_for_submission": False,
    }
    cloud.checked_training(training)
    training["strict_checkpoint_loaded"] = False
    with pytest.raises(RuntimeError, match="output"):
        cloud.checked_training(training)


def test_json_parser_rejects_non_object_output() -> None:
    completed = subprocess.CompletedProcess(["fake"], 0, stdout="[]", stderr="")
    with pytest.raises(RuntimeError, match="JSON object"):
        cloud.parse_json_output(completed, "fake verifier")
