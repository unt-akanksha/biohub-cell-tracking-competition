import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.optimize import check_grad

from research.focus_candidate_ranker import candidate_arrays, pack, combine, validate, objective, fit, metrics


def sample():
    rng = np.random.default_rng(44813)
    return dict(source_coords=np.array([[1., 2., 3.], [2., 3., 4.], [4., 2., 1.]]),
                target_coords=np.array([[1., 2., 3.], [2., 3., 4.], [8., 9., 10.], [1., 2., 3.]]),
                source_features=rng.normal(size=(3, 32)), target_features=rng.normal(size=(4, 32)),
                backward_um=np.zeros((4, 3)), labels=np.array([0, 1, 3, -1], np.int64),
                source_frame=np.array(0), target_indices=np.arange(3, 7, dtype=np.int64))


PARAMETERS = dict(mean_um=[0., 0., 0.], variance_um2=[1., 1., 1.])


def test_all_sources_plus_null_and_original_unknown_mask():
    data = pack(sample(), PARAMETERS, 'fitting')
    assert data['features'].shape == (12, 8)
    np.testing.assert_array_equal(data['sizes'], [4, 4, 4])
    np.testing.assert_array_equal(data['chosen'], [0, 5, 11])
    assert len(candidate_arrays(sample(), PARAMETERS)['starts']) == 4  # Inference has no label filtering.
    validate(data)


def test_grouped_softmax_gradient():
    data = pack(sample(), PARAMETERS, 'fitting')
    ids = validate(data)
    theta = np.zeros(8)
    assert check_grad(lambda p: objective(p, data, ids)[0], lambda p: objective(p, data, ids)[1], theta) < 2e-4


def test_roundtrip_objective_and_complete_concatenation():
    data = pack(sample(), PARAMETERS, 'fitting')
    combined = combine([data, data])
    assert metrics(combined)['loss_sum'] == pytest.approx(2 * metrics(data)['loss_sum'])
    model = fit(combined, 'fitting')
    assert model['objective'] <= model['initial_objective']
    assert metrics(combined, model) == metrics(combined, json.loads(json.dumps(model)))


def test_role_guard_before_data_access():
    with pytest.raises(ValueError, match='Only'):
        pack({}, {}, 'diagnostic')
    with pytest.raises(ValueError, match='Only'):
        fit({}, 'diagnostic')


def test_empty_source_has_one_null_choice_per_known_target():
    p = sample()
    p['source_coords'] = np.empty((0, 3))
    p['source_features'] = np.empty((0, 32))
    p['labels'] = np.array([0, 0, 0, -1], np.int64)
    data = pack(p, PARAMETERS, 'fitting')
    result = metrics(data)
    assert result['correct_absent'] == 3 and result['nll'] == 0


def test_corrupt_null_rejected():
    data = pack(sample(), PARAMETERS, 'fitting')
    data['offset'][data['null_rows'][0]] = 0
    with pytest.raises(ValueError, match='null'):
        validate(data)


def test_gate_requires_both_parent_and_absent_controls():
    root = Path(__file__).resolve().parents[1]
    gate = runpy.run_path(str(root / 'scripts/fit-focus-candidate-ranker.py'))['gate']
    physical = dict(loss_sum=14., known_parent=10, known_absent=3, correct_parent=8, correct_absent=3, nll=14/13)
    neural = dict(physical, loss_sum=13., nll=1., correct_parent=9, correct_absent=1)
    candidate = dict(neural, loss_sum=10., nll=10/13, correct_absent=2)
    folds = [dict(held_out=str(i), physical=physical, neural=neural, candidate=candidate) for i in range(12)]
    result = gate(folds)
    assert not result['passed'] and not result['conditions']['absent_not_lower']


def test_corrupt_group_coverage_rejected():
    data = pack(sample(), PARAMETERS, 'fitting')
    data['starts'][1] += 1
    with pytest.raises(ValueError, match='contiguous'):
        validate(data)
