from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_pair_appearance import projector
from research.focus_pair_appearance_head import blocks, objective
from research.focus_pair_appearance_conditioning import hessian, coordinate_map, transformed_objective

FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_focus_pair_appearance.py')))['fixture']


@pytest.mark.parametrize('arm', ['full', 'lda'])
def test_hessian_gradient_difference_and_coordinate_chain_rule(arm):
    _, base, app, stats = FIXTURE()
    projection = projector(stats, 'fitting')
    provider = lambda: blocks([(base, app)], projection, arm)
    dim = 72 if arm == 'full' else 9
    theta = np.arange(dim) * .0001
    matrix, counts = hessian(theta, provider, 2., 'fitting')
    direction = np.arange(1., dim+1)
    direction /= np.linalg.norm(direction)
    eps = 1e-5
    numeric = (objective(theta+eps*direction, provider, 2.)[1]-objective(theta-eps*direction, provider, 2.)[1])/(2*eps)
    np.testing.assert_allclose(matrix @ direction, numeric, rtol=1e-5, atol=1e-6)
    assert counts['groups'] == 3 and counts['choices'] == 12
    transform, bounds = coordinate_map(matrix, 'fitting')
    beta = np.linalg.solve(transform, theta)
    actual = transformed_objective(beta, transform, objective, provider, 2.)
    original = objective(theta, provider, 2.)
    assert actual[0] == pytest.approx(original[0], abs=1e-10)
    np.testing.assert_allclose(actual[1], transform.T @ original[1], atol=1e-8)
    for index in (4, 5, 6):
        np.testing.assert_array_equal(np.delete(transform[index], index), np.zeros(dim-1))
        assert transform[index, index] > 0
        assert bounds[index][1] * transform[index, index] <= .5
    free = [i for i in range(dim) if i not in (4, 5, 6)]
    transformed = transform.T @ matrix @ transform
    np.testing.assert_allclose(transformed[np.ix_(free, free)], np.eye(len(free)), atol=1e-8)
    np.testing.assert_allclose(transformed[np.ix_(free, [4, 5, 6])], 0, atol=1e-8)


def test_role_guards_before_data_access():
    with pytest.raises(ValueError, match='Only fitting'):
        hessian(None, None, None, 'diagnostic')
    with pytest.raises(ValueError, match='Only fitting'):
        coordinate_map(None, 'diagnostic')


def test_nonpositive_curvature_rejected_without_fallback():
    with pytest.raises(np.linalg.LinAlgError):
        coordinate_map(np.zeros((9, 9)), 'fitting')
