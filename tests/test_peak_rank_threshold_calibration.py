from __future__ import annotations

import numpy as np

from research.peak_rank_detection.calibrate_detection_threshold import (
    scored_matches,
    select_global_threshold,
)


def test_global_threshold_maximizes_micro_detection_jaccard() -> None:
    result = select_global_threshold(
        [
            (
                np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float32),
                np.asarray([True, False, True, False]),
            )
        ],
        total_truth=2,
    )
    assert result["threshold"] == np.float32(0.4)
    assert result["selected_predictions"] == 3
    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["false_negative"] == 0
    assert result["detection_jaccard"] == 2 / 3


def test_scored_matches_are_one_to_one_and_score_ordered() -> None:
    scores, matched = scored_matches(
        np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [9.0, 9.0, 9.0]]),
        np.asarray([0.8, 0.9, 0.7]),
        np.asarray([[0.0, 0.0, 0.0]]),
        radius=1.0,
    )
    np.testing.assert_allclose(scores, [0.9, 0.8, 0.7])
    np.testing.assert_array_equal(matched, [True, False, False])


def test_tied_scores_are_never_split_by_threshold() -> None:
    result = select_global_threshold(
        [
            (
                np.asarray([0.8, 0.8, 0.3], dtype=np.float32),
                np.asarray([True, False, True]),
            )
        ],
        total_truth=2,
    )
    assert result["selected_predictions"] == 3
    assert result["threshold"] < float(np.float32(0.3))
