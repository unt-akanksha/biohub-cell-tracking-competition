from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.temporal_contrastive.dual_fold_appearance_processed_acceptance import (
    verify_sources,
)


FOLDS = ("target_44b6", "target_6bba")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def source_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    trackastra_root = tmp_path / "trackastra"
    appearance_root = tmp_path / "appearance"
    trackastra_folds = {}
    calibration_folds = {}
    for index, fold in enumerate(FOLDS, start=1):
        trackastra_model = trackastra_root / fold / "model.pt"
        trackastra_model.parent.mkdir(parents=True)
        trackastra_model.write_bytes(f"trackastra-{fold}".encode())
        trackastra_fold = {
            "status": "completed",
            "fold": fold,
            "best_step": index * 100,
            "model_sha256": digest(trackastra_model),
            "submission_created": False,
        }
        trackastra_folds[fold] = trackastra_fold
        write_json(trackastra_root / fold / "worker_terminal.json", trackastra_fold)

        appearance_model = appearance_root / fold / "appearance_model.pt"
        appearance_model.parent.mkdir(parents=True)
        appearance_model.write_bytes(f"appearance-{fold}".encode())
        appearance_terminal = {
            "status": "completed",
            "fold": fold,
            "best_step": index * 200,
            "parameter_count": 19_218_498,
            "checkpoint_weight_source": "optimizer-step exponential moving average",
            "ema_decay": 0.997,
            "model_sha256": digest(appearance_model),
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        write_json(appearance_root / fold / "worker_terminal.json", appearance_terminal)
        calibration_folds[fold] = {
            "status": "completed",
            "fold": fold,
            "selection": {
                "selected_weight": 0.1 * index,
                "selected_division_weight": 0.05 * index,
                "improved": True,
            },
            "appearance_temperature": 0.10,
            "appearance_model_sha256": appearance_terminal["model_sha256"],
            "trackastra_model_sha256": trackastra_fold["model_sha256"],
            "processed_acceptance_ground_truth_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
    write_json(
        trackastra_root / "training_terminal.json",
        {
            "status": "completed",
            "gpu_count": 2,
            "both_folds_improved": True,
            "submission_created": False,
            "folds": trackastra_folds,
        },
    )
    calibration_path = tmp_path / "calibration_terminal.json"
    write_json(
        calibration_path,
        {
            "status": "completed",
            "run_id": "temporal-patch-dual-fold-blend-v1",
            "gpu_count": 2,
            "both_folds_improved": True,
            "processed_acceptance_ground_truth_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "folds": calibration_folds,
        },
    )
    return trackastra_root, appearance_root, calibration_path


def test_processed_appearance_sources_are_reciprocally_hash_bound(tmp_path: Path) -> None:
    trackastra_root, appearance_root, calibration_path = source_fixture(tmp_path)

    calibration, folds = verify_sources(
        trackastra_root, appearance_root, calibration_path
    )

    assert calibration["both_folds_improved"] is True
    assert set(folds) == set(FOLDS)
    assert folds["target_44b6"]["appearance_weight"] == 0.1
    assert folds["target_6bba"]["appearance_weight"] == 0.2
    assert folds["target_44b6"]["division_weight"] == 0.05


def test_processed_appearance_sources_reject_mutated_model(tmp_path: Path) -> None:
    trackastra_root, appearance_root, calibration_path = source_fixture(tmp_path)
    (appearance_root / "target_44b6" / "appearance_model.pt").write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="appearance fold"):
        verify_sources(trackastra_root, appearance_root, calibration_path)
