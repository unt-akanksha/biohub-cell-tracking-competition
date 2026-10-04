import json

import numpy as np
import pytest

from research.focus_pair_history_head import (FEATURES, CONSTRAINED, UPPER,
    pack, validate, objective, class_weights, blocks, hessian, coordinate_map, fit, metrics)
from research.focus_candidate_ranker import pack as base_pack, validate as base_validate


PARAMS = dict(mean_um=[0., 0., 0.], variance_um2=[1., 1., 1.])


def example():
    source = np.array([[2., 5., 8.], [5., 8., 11.]])
    packet = dict(source_frame=np.array(1, np.int64), source_coords=source,
        target_coords=np.array([[3., 7., 11.], [20., 20., 20.]]),
        source_indices=np.array([10, 11], np.int64), target_indices=np.array([20, 21], np.int64),
        source_features=np.ones((2, 32)), target_features=np.ones((2, 32)),
        backward_um=np.zeros((2, 3)), labels=np.array([0, 2], np.int64))
    previous = dict(source_frame=np.array(0, np.int64), target_coords=source.copy(),
        target_indices=np.array([10, 11], np.int64), backward_um=np.ones((2, 3)))
    return packet, previous


def test_original_objective_is_exact_subspace():
    packet, previous = example()
    old, new = base_pack(packet, PARAMS, 'fitting'), pack(packet, previous, PARAMS, 'fitting')
    ids = validate(new)
    weights, _ = class_weights(new, 'fitting')
    theta = np.arange(8)*.01
    a, ga = objective(theta, old, base_validate(old), weights)
    b, gb = objective(np.r_[theta, np.zeros(7)], new, ids, weights)
    assert a == pytest.approx(b, abs=1e-12)
    np.testing.assert_allclose(ga, gb[:8], atol=1e-12, rtol=1e-12)


def test_gradient_curvature_and_feasible_coordinates():
    packet, previous = example()
    data = pack(packet, previous, PARAMS, 'fitting')
    ids = validate(data)
    weights, aw = class_weights(data, 'fitting')
    theta = np.zeros(15)
    direction = np.arange(1., 16.)/100
    eps = 1e-6
    _, gradient = objective(theta, data, ids, weights)
    numeric = (objective(theta+eps*direction, data, ids, weights)[0]
        -objective(theta-eps*direction, data, ids, weights)[0])/(2*eps)
    assert numeric == pytest.approx(gradient @ direction, rel=2e-6, abs=2e-6)
    curvature, _ = hessian(theta, lambda: blocks(data), aw, 'fitting')
    numeric_g = (objective(theta+eps*direction, data, ids, weights)[1]
        -objective(theta-eps*direction, data, ids, weights)[1])/(2*eps)
    np.testing.assert_allclose(curvature @ direction, numeric_g, rtol=2e-6, atol=2e-6)
    transform, bounds = coordinate_map(curvature, 'fitting')
    beta = np.ones(15)
    for index in CONSTRAINED: beta[index] = bounds[index][1]
    assert np.all((transform @ beta)[CONSTRAINED] <= UPPER)
    np.testing.assert_allclose(np.linalg.solve(transform, transform @ direction), direction, atol=1e-10)


def test_guard_roles_and_null_history():
    packet, previous = example()
    with pytest.raises(ValueError): pack(packet, previous, PARAMS, 'diagnostic')
    with pytest.raises(ValueError): coordinate_map(np.eye(15), 'source')
    data = pack(packet, previous, PARAMS, 'fitting')
    data['features'][data['null_rows'][0], 8] = 1
    with pytest.raises(ValueError): validate(data)


def test_small_fit_checkpoint_reload_and_no_overwrite(tmp_path):
    packet, previous = example()
    data = pack(packet, previous, PARAMS, 'fitting')
    model, _ = fit(data, tmp_path/'fit', 'fitting')
    assert model['objective'] < model['initial_objective']
    assert model['features'] == list(FEATURES)
    restored = json.loads((tmp_path/'fit/model.json').read_text())
    assert metrics(data, model) == metrics(data, restored)
    assert (tmp_path/'fit/optimizer-terminal.json').exists()
    assert (tmp_path/'fit/coordinates.npz').exists()
    with pytest.raises(FileExistsError): fit(data, tmp_path/'fit', 'fitting')
    with pytest.raises(ValueError): fit(data, tmp_path/'not-created', 'target')
    assert not (tmp_path/'not-created').exists()
