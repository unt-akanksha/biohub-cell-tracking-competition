import numpy as np
import pytest

from research.focus_conditional_motion import FEATURES
from research.focus_conditional_motion_linking import link as constant_link
from research.focus_conditional_variance_linking import link
from research.focus_residual_calibration import VARIANCE_FLOOR


def model():
    mean = dict(features=list(FEATURES), ridge=1., center=[0.] * 6, scale=[1.] * 6,
                coefficients=np.zeros((7, 3)).tolist(), variance_um2=[1., 2., 3.])
    theta = np.zeros((7, 3))
    theta[0] = np.log(np.asarray(mean['variance_um2']) / VARIANCE_FLOOR - 1.)
    return dict(mean_model=mean, coefficients=theta.tolist(), role='fitting', ridge=1.,
                variance_floor_um2=VARIANCE_FLOOR.tolist())


def test_constant_uncertainty_reduces_to_original_graph_policy():
    coords = np.array([[t, 10. + t * .1, 20. + cell * 50, 20.]
                       for t in range(3) for cell in range(3)])
    flow = np.zeros((len(coords), 3))
    frozen = coords.copy()
    saved = model()
    actual, expected = link(coords, flow, saved), constant_link(coords, flow, saved['mean_model'])
    assert [(s, d) for s, d, _ in actual] == [(s, d) for s, d, _ in expected]
    np.testing.assert_allclose([p for _, _, p in actual], [p for _, _, p in expected], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(coords, frozen)


def test_degree_limit_empty_frame_and_no_new_nodes():
    coords = np.array([[0, 10., 20., 20.], [1, 10., 20., 20.],
                       [1, 10.1, 20., 20.], [1, 10.2, 20., 20.], [3, 10., 20., 20.]])
    edges = link(coords, np.zeros((5, 3)), model())
    assert [(s, d) for s, d, _ in edges] == [(0, 1), (0, 2)]
    assert link(np.empty((0, 4)), np.empty((0, 3)), model()) == []


def test_node_limit_rejects_instead_of_truncating():
    coords = np.tile([0., 10., 20., 20.], (2049, 1))
    with pytest.raises(ValueError, match='no node truncation'):
        link(coords, np.zeros((2049, 3)), model())


def test_invalid_coordinates_rejected():
    with pytest.raises(ValueError):
        link(np.array([[-1., 10., 20., 20.]]), np.zeros((1, 3)), model())


def test_target_dependent_axis_uncertainty_can_change_parent_ranking():
    coords = np.array([[0., 12., 20., 20.], [0., 10., 28., 20.], [1., 10., 20., 20.]])
    flow = np.zeros((3, 3))
    fitted = model()
    before = link(coords, flow, fitted)
    fitted['coefficients'][4][0] = .3  # target-Z dependence in Z uncertainty.
    after = link(coords, flow, fitted)
    assert [(s, d) for s, d, _ in before] == [(1, 2)]
    assert [(s, d) for s, d, _ in after] == [(0, 2)]
