import numpy as np
import pytest
from research.causal_motion_prior import link_causal_motion
from research.independent_motion_prior import link_motion


MODEL = dict(alpha=[.1, .45, .47], intercept_um=[-.4, .06, -.15],
             residual_variance_um2=[3.2, 1.5, 1.8])


def test_no_history_matches_frozen_prior_and_preserves_nodes():
    points = np.array([[0, 2, 2, 2], [1, 2, 3, 2], [1, 2, 3, 3]], float)
    original = points.copy()
    assert link_causal_motion(points, MODEL) == link_motion(points)
    np.testing.assert_array_equal(points, original)


def test_future_cannot_change_past_predictions():
    points = [[0, 3, 3, 3], [1, 3, 5, 3], [2, 3, 7, 3], [3, 3, 9, 3]]
    prefix = link_causal_motion(points[:3], MODEL)
    assert [(s, t, p) for s, t, p in link_causal_motion(points, MODEL) if t < 3] == prefix
    assert len(prefix) == 2


def test_division_daughter_has_no_velocity_forecast():
    points = [[0, 3, 3, 3], [1, 3, 3, 3], [1, 3, 4, 3], [2, 3, 3, 3]]
    edges = link_causal_motion(points, MODEL)
    assert len([e for e in edges if e[0] == 0]) == 2
    assert [e for e in edges if e[1] == 3] == [e for e in link_motion(points) if e[1] == 3]


def test_no_gaps_no_merges_max_two_daughters_and_null_abstention():
    assert not link_causal_motion([[0, 1, 1, 1], [2, 1, 1, 1]], MODEL)
    assert not link_causal_motion([[0, 1, 1, 1], [1, 50, 50, 50]], MODEL)
    points = [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]
    edges = link_causal_motion(points, MODEL)
    assert len(edges) == 2 and len({t for _, t, _ in edges}) == 2


def test_invalid_input_and_model_rejected():
    for points in ([[.5, 1, 1, 1]], [[0, np.nan, 1, 1]], [[0, -1, 1, 1]]):
        with pytest.raises(ValueError):
            link_causal_motion(points, MODEL)
    with pytest.raises(ValueError):
        link_causal_motion([], dict(MODEL, alpha=[2, 0, 0]))
