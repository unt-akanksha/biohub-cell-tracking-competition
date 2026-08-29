from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import research.temporal_contrastive.train_dual_fold_division_localization as trainer
from research.temporal_contrastive.train_dual_fold_division_localization import (
    AUDIT_TIMEPOINTS,
    EVALUATION_SHIFTS,
    SELECTION_TIMEPOINTS,
    baseline_metrics,
    load_division_event_patches,
    localization_gate,
)


def test_fixed_localization_evaluation_inventory_is_exhaustive_and_disjoint() -> None:
    assert len(EVALUATION_SHIFTS) == 124
    assert len(set(EVALUATION_SHIFTS)) == 124
    assert (0, 0, 0) not in EVALUATION_SHIFTS
    assert all(max(map(abs, shift)) <= 2 for shift in EVALUATION_SHIFTS)
    assert not set(SELECTION_TIMEPOINTS) & set(AUDIT_TIMEPOINTS)


def test_localization_gate_requires_mean_p90_and_every_axis() -> None:
    baseline = baseline_metrics(3)
    improved = {
        **baseline,
        "mean_residual_um": baseline["mean_residual_um"] - 0.1,
        "p90_residual_um": baseline["p90_residual_um"] - 0.1,
        "axis_mae_um": [value - 0.01 for value in baseline["axis_mae_um"]],
    }
    regressed_axis = {**improved, "axis_mae_um": [2.0, 0.5, 0.5]}

    assert localization_gate(baseline, improved)["passed"] is True
    assert localization_gate(baseline, regressed_axis)["passed"] is False


def test_event_patch_loader_keeps_parents_and_unique_daughters(tmp_path: Path) -> None:
    path = tmp_path / "shard.npz"
    sources = np.zeros((3, 3, 17, 17, 17), dtype=np.float16)
    targets = np.ones((5, 3, 17, 17, 17), dtype=np.float16)
    positive = np.zeros((3, 5), dtype=bool)
    positive[0, [1, 2]] = True
    positive[1, [2, 3]] = True
    division = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    np.savez_compressed(
        path,
        source_patches=sources,
        target_patches=targets,
        positive_mask=positive,
        division_target=division,
    )

    patches = load_division_event_patches(path, torch.device("cpu"))

    assert patches.shape == (5, 3, 17, 17, 17)
    assert torch.count_nonzero(patches[:2]) == 0
    assert torch.all(patches[2:] == 1)


def test_event_patch_loader_rejects_shards_without_divisions(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    np.savez_compressed(
        path,
        source_patches=np.zeros((1, 3, 17, 17, 17), dtype=np.float16),
        target_patches=np.zeros((2, 3, 17, 17, 17), dtype=np.float16),
        positive_mask=np.zeros((1, 2), dtype=bool),
        division_target=np.zeros(1, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="no complete division"):
        load_division_event_patches(path, torch.device("cpu"))


def test_v4_parent_is_the_accepted_external_pretraining_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = "target_44b6"
    worker = {
        "status": "completed",
        "run_id": trainer.PARENT_RUN_ID,
        "appearance_family": trainer.PARENT_FAMILY,
        "fold": fold,
        "parameter_count": trainer.PARENT_PARAMETER_COUNT,
        "best_step": 500,
        "selection_gate_passed": True,
        "audit_gate_passed": True,
        "model_sha256": "modelhash",
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": trainer.PARENT_RUN_ID,
        "appearance_family": trainer.PARENT_FAMILY,
        "gpu_count": 2,
        "both_folds_improved": True,
        "folds": {fold: worker},
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    fold_root = tmp_path / fold
    fold_root.mkdir()
    (tmp_path / trainer.PARENT_AGGREGATE_NAME).write_text(json.dumps(aggregate))
    (fold_root / "worker_terminal.json").write_text(json.dumps(worker))
    (fold_root / trainer.PARENT_MODEL_NAME).write_bytes(b"checkpoint")
    monkeypatch.setattr(
        trainer,
        "sha256_file",
        lambda path: (
            "modelhash" if path.name == trainer.PARENT_MODEL_NAME else "workerhash"
        ),
    )
    monkeypatch.setattr(trainer.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        trainer, "load_multiscale_v4_warm_start", lambda model, state: ("head.weight",)
    )

    evidence = trainer.load_v4_parent(torch.nn.Identity(), tmp_path, fold)

    assert evidence["run_id"] == "zebrahub-multiscale-contextual-pretrain-v1"
    assert evidence["parent_stage"] == "external_pretraining"
    assert evidence["model_sha256"] == "modelhash"
