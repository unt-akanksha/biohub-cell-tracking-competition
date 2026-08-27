from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-temporal-patch-runtime.py"
TARGET = ROOT / ".biohub" / "staging" / "biohub-temporal-patch-runtime-v1"
PAIR_TARGET = (
    ROOT / ".biohub" / "staging" / "biohub-temporal-pair-fusion-runtime-v2"
)
CONTEXTUAL_TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-temporal-contextual-pair-fusion-runtime-v3"
)


def test_runtime_builder_hashes_complete_two_gpu_appearance_pipeline() -> None:
    subprocess.run([sys.executable, str(BUILDER), "--replace"], check=True)
    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["integrity"] == {
        "required_gpu_count": 2,
        "opened_processed_acceptance_stems_excluded_from_training": True,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_command_included": False,
        "calibration_grid_includes_exact_zero_control": True,
        "maximum_submission_inference_seconds": 36_000,
        "minimum_kaggle_finalization_reserve_seconds": 7_200,
        "required_kaggle_machine_shape": "NvidiaTeslaT4",
        "submission_internet_enabled": False,
        "timed_out_worker_termination_grace_seconds": 15,
        "predeclared_trackastra_control_allowed": True,
    }
    required = {
        "train_dual_fold_patch.py",
        "train_dual_fold_pair_fusion.py",
        "pair_fusion.py",
        "transition_context.py",
        "contextual_pair_fusion.py",
        "contextual_training.py",
        "train_dual_fold_contextual_pair_fusion.py",
        "appearance_family.py",
        "calibrate_dual_fold_blend.py",
        "dual_fold_appearance_processed_acceptance.py",
        "dual_fold_appearance_submission.py",
        "verify_runtime.py",
        "verify_trackastra_output.py",
        "verify_appearance_output.py",
        "submission_sharding.py",
        "trackastra_source/trackastra/model/model.py",
    }
    assert required.issubset(manifest["files"])
    for name, row in manifest["files"].items():
        path = TARGET / name
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    combined = "\n".join(
        (TARGET / name).read_text(encoding="utf-8")
        for name in required
        if name.endswith(".py")
    )
    assert "device_count() != 2" in combined
    assert "kaggle competitions submit" not in combined


def test_runtime_builder_emits_distinct_pair_fusion_package() -> None:
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--replace",
            "--family",
            "pair_fusion_v2",
        ],
        check=True,
    )
    manifest = json.loads(
        (PAIR_TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (PAIR_TARGET / "experiment.json").read_text(encoding="utf-8")
    )

    assert manifest["run_id"] == "temporal-patch-pair-fusion-v2"
    assert manifest["runtime_family"] == "pair_fusion_v2"
    assert manifest["appearance_model"]["families"][
        "temporal_pair_fusion_v2"
    ]["parameters_per_fold"] == 20_869_325
    assert experiment["model"]["appearance_family"] == "temporal_pair_fusion_v2"
    assert experiment["model"]["public_code_copied"] is False
    assert (PAIR_TARGET / "train_dual_fold_pair_fusion.py").is_file()


def test_runtime_builder_emits_distinct_contextual_pair_fusion_package() -> None:
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--replace",
            "--family",
            "contextual_pair_fusion_v3",
        ],
        check=True,
    )
    manifest = json.loads(
        (CONTEXTUAL_TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (CONTEXTUAL_TARGET / "experiment.json").read_text(encoding="utf-8")
    )

    assert manifest["run_id"] == "temporal-contextual-pair-fusion-v3"
    assert manifest["runtime_family"] == "contextual_pair_fusion_v3"
    assert manifest["appearance_model"]["families"][
        "temporal_contextual_pair_fusion_v3"
    ]["parameters_per_fold"] == 20_747_761
    assert experiment["model"]["family"] == "temporal_contextual_pair_fusion_v3"
    assert experiment["public_code_copied"] is False
    assert (
        CONTEXTUAL_TARGET / "train_dual_fold_contextual_pair_fusion.py"
    ).is_file()
