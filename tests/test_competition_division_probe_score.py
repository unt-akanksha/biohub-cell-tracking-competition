from __future__ import annotations

import numpy as np
import pytest

from research.score_competition_division_probe import (
    average_precision,
    geometry_division_score,
    ranking_metrics,
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
