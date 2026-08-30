from __future__ import annotations

import numpy as np
import pytest

from research.score_competition_division_probe import (
    BIOLOGICAL_GEOMETRY_MINIMUM,
    average_precision,
    decision_metrics,
    geometry_division_score,
    ranking_metrics,
    validate_real_training_terminal,
)


def test_average_precision_rewards_positive_ranking() -> None:
    labels = np.asarray([False, True, False, True])
    scores = np.asarray([0.1, 0.9, 0.2, 0.8])

    assert average_precision(labels, scores) == pytest.approx(1.0)


def test_event_ranking_is_computed_within_each_frame() -> None:
    rows = [
        {"stem": "a", "timepoint": 3, "parent_id": 1, "safe_recovery_positive": True, "score": 0.8},
        {"stem": "a", "timepoint": 3, "parent_id": 2, "safe_recovery_positive": False, "score": 0.9},
        {"stem": "a", "timepoint": 3, "parent_id": 3, "safe_recovery_positive": False, "score": 0.1},
        {"stem": "b", "timepoint": 7, "parent_id": 4, "safe_recovery_positive": True, "score": 0.7},
        {"stem": "b", "timepoint": 7, "parent_id": 5, "safe_recovery_positive": False, "score": 0.2},
    ]

    metrics = ranking_metrics(rows, "score")

    assert [event["rank"] for event in metrics["event_ranks"]] == [2, 1]
    assert metrics["positives"] == 2
    assert metrics["precision_at_positive_count"] == pytest.approx(0.5)


def test_geometry_score_rewards_balanced_opposed_branches() -> None:
    balanced = {
        "existing_distance_um": 5.0,
        "parent_distance_um": 5.0,
        "daughter_step_ratio": 1.0,
        "daughter_opposition_cosine": -1.0,
        "daughter_midpoint_distance_um": 0.0,
    }
    ordinary = {
        "existing_distance_um": 2.0,
        "parent_distance_um": 5.0,
        "daughter_step_ratio": 0.4,
        "daughter_opposition_cosine": 0.8,
        "daughter_midpoint_distance_um": 3.0,
    }

    assert geometry_division_score(balanced) > geometry_division_score(ordinary)


def test_conjunctive_policy_requires_model_and_biological_geometry() -> None:
    rows = [
        {
            "stem": "a",
            "timepoint": 3,
            "parent_id": 1,
            "safe_recovery_positive": True,
            "ensemble_logit": 2.0,
            "geometry_division_score": 4.0,
        },
        {
            "stem": "a",
            "timepoint": 3,
            "parent_id": 2,
            "safe_recovery_positive": False,
            "ensemble_logit": 3.0,
            "geometry_division_score": 1.0,
        },
    ]

    metrics = decision_metrics(
        rows,
        model_threshold=1.5,
        geometry_minimum=BIOLOGICAL_GEOMETRY_MINIMUM,
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["precision"] == 1.0


def test_real_training_terminal_binds_frozen_models_and_threshold() -> None:
    hashes = ["a" * 64, "b" * 64]
    terminal = {
        "schema_version": 1,
        "status": "accepted_at_selection",
        "run_id": "competition-real-division-gate-v1",
        "selection_gate_passed": True,
        "final_probe_opened": False,
        "checkpoint_frozen_before_final_probe": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_final_probe": True,
        "frozen_division_logit_threshold": 1.25,
        "folds": {
            "target_44b6": {"model_sha256": hashes[0]},
            "target_6bba": {"model_sha256": hashes[1]},
        },
    }

    assert validate_real_training_terminal(terminal, hashes) == 1.25
    terminal["final_probe_opened"] = True
    with pytest.raises(ValueError):
        validate_real_training_terminal(terminal, hashes)
