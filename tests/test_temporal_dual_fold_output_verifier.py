from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.temporal_contrastive.appearance_family import COSINE_PARAMETER_COUNT
from research.temporal_contrastive.verify_dual_fold_training_output import (
    EXPECTED_SYNTHETIC_MANIFEST_SHA256,
    FOLDS,
    OPENED_ACCEPTANCE_STEMS,
    verify_output,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_output(tmp_path: Path) -> Path:
    aggregate_folds = {}
    synthetic_train = [f"synthetic_train_{index:04d}.npz" for index in range(1_900)]
    synthetic_validation = [
        f"synthetic_validation_{index:04d}.npz" for index in range(128)
    ]
    for index, (fold, spec) in enumerate(FOLDS.items(), start=1):
        fold_dir = tmp_path / fold
        fold_dir.mkdir(parents=True)
        model = fold_dir / "appearance_model.pt"
        model.write_bytes(f"appearance-{fold}".encode())
        real_composite = 0.75 + 0.01 * index
        synthetic_composite = 0.86 + 0.01 * index
        worker = {
            "schema_version": 1,
            "status": "completed",
            "run_id": "temporal-patch-dual-fold-v1",
            "fold": fold,
            "completed_step": 2_000,
            "optimizer_steps": 1_000,
            "best_step": 1_500,
            "best_score": 0.85 * real_composite + 0.15 * synthetic_composite,
            "best_real": {"top1": 0.75, "composite": real_composite},
            "best_synthetic": {"top1": 0.86, "composite": synthetic_composite},
            "model_sha256": digest(model),
            "parameter_count": COSINE_PARAMETER_COUNT,
            "input_channels": 3,
            "temporal_frame_offsets": [-1, 0, 1],
            "checkpoint_weight_source": "optimizer-step exponential moving average",
            "ema_decay": 0.997,
            "division_prior_correction": "class-conditional importance weighting",
            "link_loss_policy": "all-positive supervised contrastive mean-log-probability",
            "real_split_policy": "global deterministic disjoint partition per embryo prefix",
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        config = {
            **{
                key: worker[key]
                for key in (
                    "schema_version",
                    "run_id",
                    "parameter_count",
                    "input_channels",
                    "temporal_frame_offsets",
                    "checkpoint_weight_source",
                    "ema_decay",
                    "division_prior_correction",
                    "link_loss_policy",
                    "real_split_policy",
                    "public_predictions_copied",
                    "public_leaderboard_used_for_selection",
                    "submission_created",
                )
            },
            "fold": fold,
            "seed": spec["seed"],
            "base_channels": 64,
            "embedding_channels": 256,
            "source_prefix": spec["source_prefix"],
            "target_prefix": spec["target_prefix"],
            "opened_acceptance_stems_excluded": sorted(OPENED_ACCEPTANCE_STEMS),
            "synthetic_manifest_sha256": EXPECTED_SYNTHETIC_MANIFEST_SHA256,
            "synthetic_train_names": synthetic_train,
            "synthetic_validation_names": synthetic_validation,
            "real_train_stems": [
                f"{spec['source_prefix']}_train_{row:03d}" for row in range(96)
            ],
            "real_validation_stems": [
                f"{spec['target_prefix']}_validation_{row:03d}"
                for row in range(12)
            ],
            "real_calibration_stems_reserved": [
                f"{spec['target_prefix']}_calibration_{row:03d}"
                for row in range(12)
            ],
            "calibration_ground_truth_read": False,
            "candidate_radius_um": 32.0,
            "patch_shape": [17, 17, 17],
            "patch_half_extent_um": [8.0, 8.0, 8.0],
        }
        write_json(fold_dir / "worker_terminal.json", worker)
        write_json(fold_dir / "training_config.json", config)
        aggregate_folds[fold] = worker
    write_json(
        tmp_path / "training_terminal.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": "temporal-patch-dual-fold-v1",
            "gpu_count": 2,
            "both_folds_trained": True,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "folds": aggregate_folds,
        },
    )
    return tmp_path


def test_verifier_binds_models_and_clean_reciprocal_splits(tmp_path: Path) -> None:
    result = verify_output(valid_output(tmp_path), expected_family="temporal_cosine_v1")

    assert result["status"] == "verified"
    assert result["authorized_for_calibration"] is True
    assert result["authorized_for_submission"] is False
    assert set(result["folds"]) == set(FOLDS)


def test_verifier_rejects_worker_aggregate_divergence(tmp_path: Path) -> None:
    root = valid_output(tmp_path)
    worker_path = root / "target_44b6" / "worker_terminal.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["best_step"] += 1
    write_json(worker_path, worker)

    with pytest.raises(ValueError, match="worker diverge"):
        verify_output(root)


def test_verifier_rejects_gate_failure_and_competition_artifact(tmp_path: Path) -> None:
    root = valid_output(tmp_path)
    worker_path = root / "target_6bba" / "worker_terminal.json"
    aggregate_path = root / "training_terminal.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["best_real"]["top1"] = 0.69
    write_json(worker_path, worker)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["folds"]["target_6bba"] = deepcopy(worker)
    write_json(aggregate_path, aggregate)
    with pytest.raises(ValueError, match="checkpoint gate"):
        verify_output(root)

    worker["best_real"]["top1"] = 0.75
    write_json(worker_path, worker)
    aggregate["folds"]["target_6bba"] = deepcopy(worker)
    write_json(aggregate_path, aggregate)
    (root / "submission.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="competition artifacts"):
        verify_output(root)


def test_verifier_rejects_global_real_partition_overlap(tmp_path: Path) -> None:
    root = valid_output(tmp_path)
    config_path = root / "target_44b6" / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["real_train_stems"][0] = "6bba_validation_000"
    write_json(config_path, config)

    with pytest.raises(ValueError, match="global real partition overlaps"):
        verify_output(root)
