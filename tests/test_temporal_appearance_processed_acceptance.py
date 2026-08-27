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
    appearance_terminals = {}
    for index, fold in enumerate(FOLDS, start=1):
        trackastra_model = trackastra_root / fold / "model.pt"
        trackastra_model.parent.mkdir(parents=True)
        trackastra_model.write_bytes(f"trackastra-{fold}".encode())
        trackastra_fold = {
            "status": "completed",
            "fold": fold,
            "best_step": index * 100,
            "pretrained_initialization_retained": False,
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
            "parameter_count": 19_221_954,
            "input_channels": 3,
            "temporal_frame_offsets": [-1, 0, 1],
            "checkpoint_weight_source": "optimizer-step exponential moving average",
            "ema_decay": 0.997,
            "division_prior_correction": "class-conditional importance weighting",
            "link_loss_policy": "all-positive supervised contrastive mean-log-probability",
            "real_split_policy": "global deterministic disjoint partition per embryo prefix",
            "model_sha256": digest(appearance_model),
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        write_json(appearance_root / fold / "worker_terminal.json", appearance_terminal)
        appearance_terminals[fold] = appearance_terminal
        calibration_folds[fold] = {
            "status": "completed",
            "fold": fold,
            "selection": {
                "selected_weight": 0.1 * index,
                "selected_division_weight": 0.05 * index,
                "selected_ensemble_mode": "reciprocal_mean",
                "improved": True,
            },
            "appearance_temperature": 0.10,
            "appearance_model_sha256": appearance_terminal["model_sha256"],
            "trackastra_model_sha256": trackastra_fold["model_sha256"],
            "trackastra_source_policy": "adapted_dual_fold",
            "processed_acceptance_ground_truth_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
    for fold in FOLDS:
        peer_fold = next(candidate for candidate in FOLDS if candidate != fold)
        calibration_folds[fold]["peer_fold"] = peer_fold
        calibration_folds[fold]["peer_appearance_model_sha256"] = (
            appearance_terminals[peer_fold]["model_sha256"]
        )
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
    assert folds["target_44b6"]["ensemble_mode"] == "reciprocal_mean"


def test_processed_appearance_sources_reject_mutated_model(tmp_path: Path) -> None:
    trackastra_root, appearance_root, calibration_path = source_fixture(tmp_path)
    (appearance_root / "target_44b6" / "appearance_model.pt").write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="appearance fold"):
        verify_sources(trackastra_root, appearance_root, calibration_path)


def test_processed_appearance_sources_reject_worker_aggregate_divergence(
    tmp_path: Path,
) -> None:
    trackastra_root, appearance_root, calibration_path = source_fixture(tmp_path)
    worker_path = trackastra_root / "target_44b6" / "worker_terminal.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["best_step"] += 1
    write_json(worker_path, worker)

    with pytest.raises(RuntimeError, match="worker/aggregate mismatch"):
        verify_sources(trackastra_root, appearance_root, calibration_path)


def test_processed_appearance_sources_accept_identical_predeclared_control(
    tmp_path: Path,
) -> None:
    trackastra_root, appearance_root, calibration_path = source_fixture(tmp_path)
    aggregate_path = trackastra_root / "training_terminal.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["both_folds_improved"] = False
    control_bytes = b"one-predeclared-trackastra-control"
    for fold in FOLDS:
        model_path = trackastra_root / fold / "model.pt"
        model_path.write_bytes(control_bytes)
        worker = aggregate["folds"][fold]
        initial_real = {"composite": 0.999}
        initial_synthetic = {"composite": 0.92}
        worker.update(
            {
                "best_step": 0,
                "pretrained_initialization_retained": True,
                "initial_real": initial_real,
                "best_real": initial_real,
                "initial_synthetic": initial_synthetic,
                "best_synthetic": initial_synthetic,
                "initial_selection_score": 0.98,
                "best_selection_score": 0.98,
                "model_sha256": digest(model_path),
            }
        )
        write_json(trackastra_root / fold / "worker_terminal.json", worker)
    write_json(aggregate_path, aggregate)

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    for fold in FOLDS:
        calibration["folds"][fold]["trackastra_model_sha256"] = digest(
            trackastra_root / fold / "model.pt"
        )
        calibration["folds"][fold]["trackastra_source_policy"] = (
            "predeclared_pretrained_control"
        )
    write_json(calibration_path, calibration)

    _calibration, folds = verify_sources(
        trackastra_root, appearance_root, calibration_path
    )

    assert {
        row["trackastra_source_policy"] for row in folds.values()
    } == {"predeclared_pretrained_control"}
