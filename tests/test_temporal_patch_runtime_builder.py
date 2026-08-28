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
TRANSFER_TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-temporal-contextual-transfer-runtime-v1"
)
MULTISCALE_TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-temporal-multiscale-contextual-runtime-v4"
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
        "contextual_v3_requires_hash_bound_external_pretraining": True,
        "multiscale_v4_requires_accepted_contextual_v3_warm_start": True,
    }
    required = {
        "train_dual_fold_patch.py",
        "train_dual_fold_pair_fusion.py",
        "pair_fusion.py",
        "transition_context.py",
        "contextual_pair_fusion.py",
        "contextual_training.py",
        "train_dual_fold_contextual_pair_fusion.py",
        "train_zebrahub_contextual_pretrain.py",
        "verify_zebrahub_contextual_dataset.py",
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
    assert manifest["appearance_model"]["families"][
        "temporal_contextual_pair_fusion_v3"
    ]["reciprocal_parent_loss_weight"] == 0.35
    assert manifest["appearance_model"]["families"][
        "temporal_contextual_pair_fusion_v3"
    ]["external_validation_precision"] == "CUDA float16 autocast"
    assert manifest["appearance_model"]["families"][
        "temporal_contextual_pair_fusion_v3"
    ]["candidate_pair_inference_precision"] == "CUDA float16 autocast"
    assert manifest["appearance_model"]["families"][
        "temporal_contextual_pair_fusion_v3"
    ]["minimum_external_composite_gain"] == 0.01
    assert manifest["appearance_model"][
        "external_pretrained_weights_required_by_family"
    ]["temporal_contextual_pair_fusion_v3"] is True
    assert experiment["model"]["family"] == "temporal_contextual_pair_fusion_v3"
    assert experiment["public_code_copied"] is False
    assert experiment["pretraining_execution"]["status"] == "runtime_package"
    assert (
        experiment["pretraining_execution"]["runtime_manifest_sha256"]
        == "see_SOURCE_MANIFEST.json"
    )
    assert (
        experiment["pretraining_execution"]["runtime_remote_redownload_verified"]
        is False
    )
    assert experiment["external_data"]["remote_redownload_verified"] is False
    assert (
        experiment["pretraining_execution"]["kernel_notebook_sha256"]
        == "recorded_outside_runtime_package"
    )
    assert (
        experiment["source"]["zebrahub_pretraining_kernel_builder_sha256"]
        == "recorded_outside_runtime_package"
    )
    assert (
        experiment["source"]["technique_report_sha256"]
        == "recorded_outside_runtime_package"
    )
    assert (
        CONTEXTUAL_TARGET / "train_dual_fold_contextual_pair_fusion.py"
    ).is_file()


def test_runtime_builder_emits_gated_contextual_transfer_package() -> None:
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--replace",
            "--family",
            "contextual_transfer_v3",
        ],
        check=True,
    )
    manifest = json.loads(
        (TRANSFER_TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (TRANSFER_TARGET / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (TRANSFER_TARGET / "experiment.json").read_text(encoding="utf-8")
    )
    trainer = (TRANSFER_TARGET / "train_dual_fold_pair_fusion.py").read_text(
        encoding="utf-8"
    )
    verifier = (TRANSFER_TARGET / "verify_appearance_output.py").read_text(
        encoding="utf-8"
    )

    assert manifest["run_id"] == "temporal-contextual-pair-fusion-v3"
    assert manifest["runtime_family"] == "contextual_transfer_v3"
    assert metadata["id"] == (
        "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1"
    )
    assert metadata["isPrivate"] is True
    assert "def finetuning_improvement_gate" in trainer
    assert "MINIMUM_REAL_COMPOSITE_GAIN = 0.005" in trainer
    assert "MAXIMUM_SYNTHETIC_METRIC_REGRESSION = 0.01" in trainer
    assert "verify_finetuning_gate" in verifier
    assert 'aggregate.get("both_folds_improved") is True' in verifier
    assert experiment["transfer_execution"]["status"] == "runtime_package"
    assert (
        experiment["transfer_execution"]["runtime_manifest_sha256"]
        == "see_SOURCE_MANIFEST.json"
    )
    assert (
        experiment["transfer_execution"]["runtime_dataset_version"]
        == "recorded_outside_runtime_package"
    )
    assert set(experiment["source"].values()) == {
        "recorded_outside_runtime_package"
    }


def test_runtime_builder_emits_high_capacity_multiscale_v4_package() -> None:
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--replace",
            "--family",
            "multiscale_contextual_v4",
        ],
        check=True,
    )
    manifest = json.loads(
        (MULTISCALE_TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (MULTISCALE_TARGET / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (MULTISCALE_TARGET / "experiment.json").read_text(encoding="utf-8")
    )

    family = manifest["appearance_model"]["families"][
        "temporal_multiscale_contextual_pair_fusion_v4"
    ]
    assert manifest["run_id"] == "temporal-multiscale-contextual-pair-fusion-v4"
    assert manifest["runtime_family"] == "multiscale_contextual_v4"
    assert family["parameters_per_fold"] == 46_386_607
    assert family["projection_base_channels"] == 96
    assert family["zero_residual_warm_start_preserves_v3_predictions"] is True
    assert metadata["id"] == (
        "indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4"
    )
    assert metadata["isPrivate"] is True
    assert experiment["model"]["family"] == (
        "temporal_multiscale_contextual_pair_fusion_v4"
    )
    assert experiment["model"]["parameter_count_per_fold"] == 46_386_607
    assert experiment["pretraining_execution"]["status"] == "runtime_package"
    assert experiment["pretraining_execution"]["runtime_dataset_version"] == (
        "recorded_outside_runtime_package"
    )
    assert experiment["pretraining_execution"][
        "excluded_runtime_dataset_versions"
    ] == "recorded_outside_runtime_package"
    assert experiment["transfer_execution"]["status"] == "runtime_package"
    assert set(experiment["source"].values()) == {
        "recorded_outside_runtime_package"
    }
    for name in (
        "multiscale_contextual_pair_fusion.py",
        "train_zebrahub_multiscale_contextual_pretrain.py",
        "train_dual_fold_multiscale_contextual_pair_fusion.py",
    ):
        assert name in manifest["files"]
        assert (MULTISCALE_TARGET / name).is_file()
