from __future__ import annotations

import numpy as np
import pytest

from research.lsm_fm_detection.evaluate_ensemble import (
    CANDIDATES,
    REFERENCE_CANDIDATE,
    combine_probabilities,
    select_global_candidate,
)


def summary(pooled: float, rows: dict[str, float], distance: float = 2.0) -> dict:
    return {
        "annotated_node_recall": pooled,
        "worst_movie_recall": min(rows.values()),
        "mean_movie_match_distance_um": distance,
        "rows": [
            {"stem": stem, "candidate": {"annotated_node_recall": recall}}
            for stem, recall in rows.items()
        ],
    }


def test_equal_ensemble_is_exact_probability_mean() -> None:
    first = np.full((2, 2, 2), 0.2, dtype=np.float32)
    second = np.full((2, 2, 2), 0.8, dtype=np.float32)
    candidate = next(value for value in CANDIDATES if value.name == "equal_ensemble_control")
    combined = combine_probabilities(first, second, candidate)
    np.testing.assert_allclose(combined, 0.5)


def test_invalid_ensemble_weights_fail_closed() -> None:
    candidate = type(CANDIDATES[0])("invalid", 0.0, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        combine_probabilities(np.ones((2, 2, 2)), np.ones((2, 2, 2)), candidate)


def test_selection_prefers_global_ensemble_worst_movie_gain() -> None:
    summaries = {
        "feature24_control": summary(0.88, {"a": 0.89, "hard": 0.63}),
        REFERENCE_CANDIDATE: summary(0.89, {"a": 0.90, "hard": 0.62}),
        "feature36_log_quadratic": summary(0.891, {"a": 0.90, "hard": 0.64}),
        "equal_ensemble_control": summary(0.90, {"a": 0.895, "hard": 0.68}),
        "equal_ensemble_log_quadratic": summary(0.89, {"a": 0.87, "hard": 0.70}),
    }
    selected, diagnostics = select_global_candidate(summaries)
    assert selected == "equal_ensemble_control"
    assert diagnostics["equal_ensemble_control"]["selection_passed"] is True
    assert diagnostics["equal_ensemble_log_quadratic"]["selection_passed"] is False
