from __future__ import annotations

import numpy as np
import pytest

from research.temporal_contrastive.future_division_context import (
    FUTURE_DIVISION_FEATURE_WIDTH,
    future_division_evidence,
)


def test_future_division_features_capture_distinct_outward_continuations() -> None:
    daughters = np.asarray([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    future = np.asarray(
        [[0.0, -3.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]
    )
    scores = np.asarray([[0.95, 0.10, 0.30], [0.05, 0.90, 0.25]])
    evidence = future_division_evidence(
        daughters, future, scores, np.ones_like(scores, dtype=bool)
    )
    features = evidence.feature_vector(candidate_radius_um=10.0)

    assert evidence.available is True
    assert evidence.continuation_indices == (0, 1)
    assert evidence.current_separation_um == pytest.approx(2.0)
    assert evidence.future_separation_um == pytest.approx(6.0)
    assert evidence.separation_gain_um == pytest.approx(4.0)
    assert evidence.assignment_margin == pytest.approx(1.70)
    assert evidence.separation_direction_cosine == pytest.approx(1.0)
    assert features.shape == (FUTURE_DIVISION_FEATURE_WIDTH,)
    assert np.isfinite(features).all()


def test_future_division_features_are_daughter_permutation_invariant() -> None:
    daughters = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    future = np.asarray([[2.5, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    scores = np.asarray([[0.88, 0.04, 0.2], [0.03, 0.91, 0.1]])
    mask = np.ones_like(scores, dtype=bool)
    direct = future_division_evidence(daughters, future, scores, mask)
    swapped = future_division_evidence(
        daughters[::-1], future, scores[::-1], mask[::-1]
    )

    assert direct.available and swapped.available
    assert direct.continuation_indices == tuple(reversed(swapped.continuation_indices))
    assert direct.feature_vector(candidate_radius_um=8.0) == pytest.approx(
        swapped.feature_vector(candidate_radius_um=8.0)
    )


def test_future_division_features_report_convergence_without_deciding() -> None:
    daughters = np.asarray([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]])
    future = np.asarray([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])
    scores = np.asarray([[0.8, 0.1], [0.2, 0.85]])
    evidence = future_division_evidence(
        daughters, future, scores, np.ones_like(scores, dtype=bool)
    )

    assert evidence.available is True
    assert evidence.separation_gain_um == pytest.approx(-3.0)
    assert evidence.feature_vector(candidate_radius_um=10.0)[2] == pytest.approx(-0.3)


def test_future_division_features_fail_closed_without_distinct_continuations() -> None:
    daughters = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    future = np.asarray([[0.0, 2.0, 0.0]])
    scores = np.asarray([[0.8], [0.9]])
    evidence = future_division_evidence(
        daughters, future, scores, np.ones_like(scores, dtype=bool)
    )

    assert evidence.available is False
    assert evidence.continuation_indices is None
    assert np.array_equal(
        evidence.feature_vector(candidate_radius_um=10.0),
        np.zeros(FUTURE_DIVISION_FEATURE_WIDTH, dtype=np.float32),
    )


def test_future_division_features_reject_nonfinite_eligible_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        future_division_evidence(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.asarray([[0.8, np.nan], [0.7, 0.9]]),
            np.ones((2, 2), dtype=bool),
        )
