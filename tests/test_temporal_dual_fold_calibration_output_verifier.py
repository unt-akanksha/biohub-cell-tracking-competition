from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.temporal_contrastive.verify_dual_fold_calibration_output import (
    APPEARANCE_FAMILY,
    APPEARANCE_RUN_ID,
    APPEARANCE_TEMPERATURE,
    APPEARANCE_WEIGHTS,
    DIVISION_WEIGHTS,
    ENSEMBLE_MODES,
    FOLDS,
    FROZEN_LINK_CONFIGURATION,
    MAXIMUM_MOVIE_REGRESSION,
    MINIMUM_POOLED_GAIN,
    RUN_ID,
    TRACKASTRA_RUN_ID,
    verify_output,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_row(stem: str | None, *, improved: bool, movies: int = 1) -> dict:
    edge_tp = (81 if improved else 80) * movies
    counts = {
        "edge_tp": edge_tp,
        "edge_fp": (9 if improved else 10) * movies,
        "edge_fn": 10 * movies,
        "division_tp": 8 * movies,
        "division_fp": 1 * movies,
        "division_fn": 1 * movies,
        "predicted_edges": (90 if improved else 90) * movies,
        "true_edges": (91 if improved else 90) * movies,
    }
    edge = counts["edge_tp"] / (
        counts["edge_tp"] + counts["edge_fp"] + counts["edge_fn"]
    )
    division = counts["division_tp"] / (
        counts["division_tp"]
        + counts["division_fp"]
        + counts["division_fn"]
    )
    row = {
        **counts,
        "edge_jaccard": edge,
        "adjusted_edge_jaccard": edge,
        "division_jaccard": division,
        "composite": edge + 0.10 * division,
    }
    if stem is not None:
        row["stem"] = stem
    else:
        row["movies"] = movies
    return row


def selection_fixture(stems: list[str]) -> dict:
    control_movies = [metric_row(stem, improved=False) for stem in stems]
    control_composite = metric_row(None, improved=False, movies=len(stems))["composite"]
    grid = []
    for mode in ENSEMBLE_MODES:
        for appearance_weight in APPEARANCE_WEIGHTS:
            for division_weight in DIVISION_WEIGHTS:
                improved = appearance_weight > 0
                by_movie = [metric_row(stem, improved=improved) for stem in stems]
                pooled = metric_row(None, improved=improved, movies=len(stems))
                deltas = {
                    stem: row["composite"] - control["composite"]
                    for stem, row, control in zip(stems, by_movie, control_movies)
                }
                pooled_gain = pooled["composite"] - control_composite
                worst_delta = min(deltas.values())
                eligible = bool(
                    appearance_weight > 0
                    and pooled_gain >= MINIMUM_POOLED_GAIN
                    and worst_delta >= -MAXIMUM_MOVIE_REGRESSION
                )
                grid.append(
                    {
                        "ensemble_mode": mode,
                        "appearance_weight": appearance_weight,
                        "division_weight": division_weight,
                        "by_movie": by_movie,
                        "pooled": pooled,
                        "pooled_gain_vs_zero": pooled_gain,
                        "worst_movie_delta_vs_zero": worst_delta,
                        "movie_deltas_vs_zero": deltas,
                        "eligible": eligible,
                    }
                )
    control = next(
        row
        for row in grid
        if (row["ensemble_mode"], row["appearance_weight"], row["division_weight"])
        == ("target_only", 0.0, 0.0)
    )
    selected = next(
        row
        for row in grid
        if (row["ensemble_mode"], row["appearance_weight"], row["division_weight"])
        == ("target_only", 0.05, 0.0)
    )
    return {
        "selected_weight": 0.05,
        "selected_division_weight": 0.0,
        "selected_ensemble_mode": "target_only",
        "improved": True,
        "selected": deepcopy(selected),
        "control": deepcopy(control),
        "grid": grid,
    }


def valid_sources(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    appearance_root = tmp_path / "appearance"
    trackastra_root = tmp_path / "trackastra"
    appearance_folds = {}
    trackastra_folds = {}
    for fold in FOLDS:
        prefix = "44b6" if fold == "target_44b6" else "6bba"
        stems = [f"{prefix}_calibration_{index:02d}" for index in range(12)]

        appearance_model = appearance_root / fold / "appearance_model.pt"
        appearance_model.parent.mkdir(parents=True)
        appearance_model.write_bytes(f"appearance-{fold}".encode())
        appearance_worker = {
            "schema_version": 1,
            "status": "completed",
            "run_id": APPEARANCE_RUN_ID,
            "fold": fold,
            "model_sha256": digest(appearance_model),
            "appearance_family": APPEARANCE_FAMILY,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        write_json(appearance_root / fold / "worker_terminal.json", appearance_worker)
        write_json(
            appearance_root / fold / "training_config.json",
            {"real_calibration_stems_reserved": stems},
        )
        appearance_folds[fold] = appearance_worker

        trackastra_model = trackastra_root / fold / "model.pt"
        trackastra_model.parent.mkdir(parents=True)
        trackastra_model.write_bytes(b"shared-trackastra-control")
        real = {"composite": 0.5}
        synthetic = {"composite": 0.6}
        trackastra_worker = {
            "schema_version": 1,
            "status": "completed",
            "run_id": TRACKASTRA_RUN_ID,
            "fold": fold,
            "best_step": 0,
            "pretrained_initialization_retained": True,
            "initial_real": real,
            "best_real": real,
            "initial_synthetic": synthetic,
            "best_synthetic": synthetic,
            "initial_selection_score": 0.51,
            "best_selection_score": 0.51,
            "model_sha256": digest(trackastra_model),
            "submission_created": False,
        }
        write_json(trackastra_root / fold / "worker_terminal.json", trackastra_worker)
        trackastra_folds[fold] = trackastra_worker

    appearance = {
        "schema_version": 1,
        "status": "completed",
        "run_id": APPEARANCE_RUN_ID,
        "appearance_family": APPEARANCE_FAMILY,
        "gpu_count": 2,
        "both_folds_trained": True,
        "both_folds_improved": True,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "folds": appearance_folds,
    }
    trackastra = {
        "schema_version": 1,
        "status": "completed",
        "run_id": TRACKASTRA_RUN_ID,
        "gpu_count": 2,
        "whole_fold_coverage": list(FOLDS),
        "both_folds_improved": False,
        "submission_created": False,
        "folds": trackastra_folds,
    }
    write_json(appearance_root / "training_terminal.json", appearance)
    write_json(trackastra_root / "training_terminal.json", trackastra)
    return appearance_root, trackastra_root, appearance, trackastra


def valid_output(tmp_path: Path) -> tuple[Path, Path, Path]:
    appearance_root, trackastra_root, appearance, trackastra = valid_sources(tmp_path)
    root = tmp_path / "download"
    calibration_root = root / "temporal_contextual_calibration_v3"
    folds = {}
    for fold in FOLDS:
        peer_fold = next(candidate for candidate in FOLDS if candidate != fold)
        prefix = "44b6" if fold == "target_44b6" else "6bba"
        stems = [f"{prefix}_calibration_{index:02d}" for index in range(12)]
        row = {
            "schema_version": 1,
            "status": "completed",
            "run_id": RUN_ID,
            "appearance_family": APPEARANCE_FAMILY,
            "fold": fold,
            "calibration_stems": stems,
            "selection": selection_fixture(stems),
            "appearance_model_sha256": appearance["folds"][fold]["model_sha256"],
            "peer_fold": peer_fold,
            "peer_appearance_model_sha256": appearance["folds"][peer_fold]["model_sha256"],
            "trackastra_model_sha256": trackastra["folds"][fold]["model_sha256"],
            "trackastra_source_policy": "predeclared_pretrained_control",
            "ensemble_mode_grid": list(ENSEMBLE_MODES),
            "appearance_weight_grid": list(APPEARANCE_WEIGHTS),
            "division_weight_grid": list(DIVISION_WEIGHTS),
            "appearance_temperature": APPEARANCE_TEMPERATURE,
            "frozen_clean_link_configuration": FROZEN_LINK_CONFIGURATION,
            "minimum_pooled_gain": MINIMUM_POOLED_GAIN,
            "maximum_movie_regression": MAXIMUM_MOVIE_REGRESSION,
            "processed_acceptance_ground_truth_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        write_json(calibration_root / fold / "calibration_result.json", row)
        folds[fold] = row
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "appearance_family": APPEARANCE_FAMILY,
        "gpu_count": 2,
        "both_folds_improved": True,
        "folds": folds,
        "processed_acceptance_ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    aggregate_path = calibration_root / "calibration_terminal.json"
    write_json(aggregate_path, aggregate)
    write_json(
        root / "calibration_launcher_terminal.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": RUN_ID,
            "declared_budget_seconds": 21_600,
            "calibrator_hard_stop_seconds": 19_800,
            "calibration_terminal_exists": True,
            "calibration_terminal_sha256": digest(aggregate_path),
            "gpu_count_required": 2,
            "processed_acceptance_ground_truth_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )
    return root, appearance_root, trackastra_root


def test_verifier_recomputes_grid_and_authorizes_only_processed(tmp_path: Path) -> None:
    root, appearance_root, trackastra_root = valid_output(tmp_path)

    result = verify_output(
        root, appearance_root=appearance_root, trackastra_root=trackastra_root
    )

    assert result["status"] == "verified"
    assert result["authorized_for_processed_materialization"] is True
    assert result["authorized_for_submission"] is False
    assert all(row["appearance_weight"] == 0.05 for row in result["selected"].values())


def test_verifier_rejects_launcher_hash_divergence(tmp_path: Path) -> None:
    root, appearance_root, trackastra_root = valid_output(tmp_path)
    launcher_path = root / "calibration_launcher_terminal.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    launcher["calibration_terminal_sha256"] = "0" * 64
    write_json(launcher_path, launcher)

    with pytest.raises(ValueError, match="launcher terminal"):
        verify_output(root, appearance_root=appearance_root, trackastra_root=trackastra_root)


def test_verifier_rejects_incomplete_grid_even_when_flags_pass(tmp_path: Path) -> None:
    root, appearance_root, trackastra_root = valid_output(tmp_path)
    aggregate_path = root / "temporal_contextual_calibration_v3/calibration_terminal.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["folds"]["target_44b6"]["selection"]["grid"].pop()
    write_json(
        root / "temporal_contextual_calibration_v3/target_44b6/calibration_result.json",
        aggregate["folds"]["target_44b6"],
    )
    write_json(aggregate_path, aggregate)
    launcher_path = root / "calibration_launcher_terminal.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    launcher["calibration_terminal_sha256"] = digest(aggregate_path)
    write_json(launcher_path, launcher)

    with pytest.raises(ValueError, match="40 rows"):
        verify_output(root, appearance_root=appearance_root, trackastra_root=trackastra_root)


def test_verifier_rejects_selected_row_that_is_not_recomputed_winner(tmp_path: Path) -> None:
    root, appearance_root, trackastra_root = valid_output(tmp_path)
    aggregate_path = root / "temporal_contextual_calibration_v3/calibration_terminal.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    selection = aggregate["folds"]["target_6bba"]["selection"]
    wrong = next(
        row
        for row in selection["grid"]
        if (row["ensemble_mode"], row["appearance_weight"], row["division_weight"])
        == ("reciprocal_mean", 0.35, 0.20)
    )
    selection["selected"] = deepcopy(wrong)
    selection["selected_weight"] = 0.35
    selection["selected_division_weight"] = 0.20
    selection["selected_ensemble_mode"] = "reciprocal_mean"
    write_json(
        root / "temporal_contextual_calibration_v3/target_6bba/calibration_result.json",
        aggregate["folds"]["target_6bba"],
    )
    write_json(aggregate_path, aggregate)
    launcher_path = root / "calibration_launcher_terminal.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    launcher["calibration_terminal_sha256"] = digest(aggregate_path)
    write_json(launcher_path, launcher)

    with pytest.raises(ValueError, match="not the grid winner"):
        verify_output(root, appearance_root=appearance_root, trackastra_root=trackastra_root)


def test_verifier_rejects_source_checkpoint_hash_divergence(tmp_path: Path) -> None:
    root, appearance_root, trackastra_root = valid_output(tmp_path)
    (appearance_root / "target_44b6/appearance_model.pt").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        verify_output(root, appearance_root=appearance_root, trackastra_root=trackastra_root)
