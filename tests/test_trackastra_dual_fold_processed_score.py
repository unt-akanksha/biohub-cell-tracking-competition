from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest

from research.trackastra_graph.dual_fold_processed_acceptance import (
    FROZEN_ASSOCIATION_CONFIGURATION,
    configuration_sha256,
)
from research.trackastra_graph.score_dual_fold_processed_candidate import (
    EXPECTED_CONTROL_SCORE,
    assert_identical_nodes,
    exact_gate,
    validate_materialization,
)


def materialization() -> dict:
    return {
        "status": "completed",
        "evaluation_kind": "predeclared_processed_candidate_materialization",
        "gpu_count": 2,
        "whole_movie_sharding": True,
        "ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
        "exact_processed_scoring_performed": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
        "processed_control_sha256": "a" * 64,
        "processed_candidate_sha256": "b" * 64,
        "association_configuration": FROZEN_ASSOCIATION_CONFIGURATION,
        "association_configuration_sha256": configuration_sha256(
            FROZEN_ASSOCIATION_CONFIGURATION
        ),
        "models": {
            "target_44b6": {"model_sha256": "c" * 64, "best_step": 10},
            "target_6bba": {"model_sha256": "d" * 64, "best_step": 20},
        },
    }


def test_materialization_must_be_hash_bound_and_selection_safe() -> None:
    validate_materialization(
        materialization(), control_sha256="a" * 64, candidate_sha256="b" * 64
    )
    unsafe = deepcopy(materialization())
    unsafe["public_leaderboard_used_for_selection"] = True
    with pytest.raises(ValueError, match="not selection-safe"):
        validate_materialization(
            unsafe, control_sha256="a" * 64, candidate_sha256="b" * 64
        )


def test_appearance_materialization_requires_two_positive_hash_bound_blends() -> None:
    payload = materialization()
    payload.update(
        {
            "candidate_family": "trackastra_appearance_blend",
            "calibration_terminal_sha256": "e" * 64,
            "appearance_models": {
                "target_44b6": {
                    "model_sha256": "f" * 64,
                    "best_step": 30,
                    "checkpoint_weight_source": "optimizer-step exponential moving average",
                    "ema_decay": 0.997,
                },
                "target_6bba": {
                    "model_sha256": "1" * 64,
                    "best_step": 40,
                    "checkpoint_weight_source": "optimizer-step exponential moving average",
                    "ema_decay": 0.997,
                },
            },
            "appearance_blend": {
                "target_44b6": {
                    "appearance_weight": 0.10,
                    "division_weight": 0.05,
                    "appearance_temperature": 0.10,
                },
                "target_6bba": {
                    "appearance_weight": 0.20,
                    "division_weight": 0.10,
                    "appearance_temperature": 0.10,
                },
            },
        }
    )
    validate_materialization(
        payload, control_sha256="a" * 64, candidate_sha256="b" * 64
    )
    payload["appearance_blend"]["target_44b6"]["appearance_weight"] = 0.0
    with pytest.raises(ValueError, match="blend evidence"):
        validate_materialization(
            payload, control_sha256="a" * 64, candidate_sha256="b" * 64
        )


def test_node_identity_ignores_csv_row_id_but_not_coordinates() -> None:
    schema = {
        "id": pl.Int64,
        "dataset": pl.String,
        "row_type": pl.String,
        "node_id": pl.Int64,
        "t": pl.Int64,
        "z": pl.Int64,
        "y": pl.Int64,
        "x": pl.Int64,
        "source_id": pl.Int64,
        "target_id": pl.Int64,
    }
    control = pl.DataFrame(
        [(0, "44b6_a", "node", 7, 0, 1, 2, 3, -1, -1)], schema=schema, orient="row"
    )
    candidate = control.with_columns(pl.lit(999).alias("id"))
    assert assert_identical_nodes(control, candidate) == 1
    changed = candidate.with_columns(pl.lit(4).alias("x"))
    with pytest.raises(RuntimeError, match="changed node"):
        assert_identical_nodes(control, changed)


def test_exact_gate_requires_pooled_gain_and_bounded_movie_regression() -> None:
    control = {"score": EXPECTED_CONTROL_SCORE, "node_recall_micro": 0.98}
    candidate = {"score": EXPECTED_CONTROL_SCORE + 0.001, "node_recall_micro": 0.98}
    by_movie = [
        {"sample_id": "a", "score_delta": -0.0019, "node_recall_delta": 0.0},
        {"sample_id": "b", "score_delta": 0.0039, "node_recall_delta": 0.0},
    ]
    assert exact_gate(control, candidate, by_movie, edge_sets_differ=True)["passed"]

    regressed = deepcopy(by_movie)
    regressed[0]["score_delta"] = -0.0021
    rejected = exact_gate(control, candidate, regressed, edge_sets_differ=True)
    assert rejected["passed"] is False
    assert rejected["checks"]["per_movie_regression_floor_passed"] is False
