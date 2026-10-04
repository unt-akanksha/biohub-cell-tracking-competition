import json

import numpy as np
import pytest
from scipy.optimize import check_grad

from research.focus_candidate_ranker import pack, combine, validate, metrics, objective as unweighted
from research.focus_balanced_candidate_ranker import class_weights, objective, fit, frame_packet, error_counts


def data():
    rng = np.random.default_rng(274919)
    packet = dict(source_coords=np.array([[0., 0., 0.], [1., 2., 3.]]),
                  target_coords=np.array([[0., 0., 0.], [1., 2., 3.], [7., 8., 9.]]),
                  source_features=rng.normal(size=(2, 32)), target_features=rng.normal(size=(3, 32)),
                  backward_um=np.zeros((3, 3)), labels=np.array([0, 1, 2], np.int64),
                  source_frame=np.array(0), target_indices=np.arange(2, 5, dtype=np.int64))
    return pack(packet, dict(mean_um=[0., 0., 0.], variance_um2=[1., 1., 1.]), 'fitting')


def test_training_counts_and_weight():
    weights, value = class_weights(data(), 'fitting')
    assert value == np.sqrt(2)
    np.testing.assert_array_equal(weights, [1., 1., np.sqrt(2)])


def test_weighted_gradient_and_unweighted_reduction():
    sample = data()
    ids = validate(sample)
    weights, _ = class_weights(sample, 'fitting')
    theta = np.array([.2, .01, -.02, .03, -.01, -.02, -.03, .4])
    assert check_grad(lambda p: objective(p, sample, ids, weights)[0],
                      lambda p: objective(p, sample, ids, weights)[1], theta) < 3e-4
    left = objective(theta, sample, ids, np.ones(3))
    right = unweighted(theta, sample, ids)
    assert left[0] == right[0]
    np.testing.assert_allclose(left[1], right[1], atol=1e-12)


def test_roles_rejected_before_data_access():
    with pytest.raises(ValueError, match='Only fitting'):
        fit({}, 'diagnostic')
    with pytest.raises(ValueError, match='Only fitting'):
        class_weights({}, 'diagnostic')


def test_no_silent_weight_fallback():
    with pytest.raises(ValueError, match='Both fitting'):
        class_weights(dict(present=np.ones(3, dtype=np.int64)), 'fitting')
    with pytest.raises(ValueError, match='Exact binary'):
        class_weights(dict(present=np.array([1, -1], dtype=np.int64)), 'fitting')


def test_real_parameter_reload_and_errors_cover_parents():
    sample = data()
    model = fit(sample, 'fitting')
    restored = json.loads(json.dumps(model))
    result = metrics(sample, model)
    assert result == metrics(sample, restored)
    assert model['objective'] <= model['initial_objective']
    errors = error_counts(sample, model)
    assert result['correct_parent'] + errors['parent_rejected'] + errors['parent_wrong_source'] == 2
    assert errors['absent_linked'] + result['correct_absent'] == 1
    assert errors['parent_ranking_ceiling'] >= result['correct_parent']


def test_full_frame_slice_keeps_all_candidates_and_original_labels():
    first, second = data(), data()
    second['source_frame'][:] = 1
    joined = combine([first, second])
    for frame, expected in enumerate((first, second)):
        actual = frame_packet(joined, frame)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    with pytest.raises(ValueError, match='complete contiguous'):
        frame_packet(joined, 2)
