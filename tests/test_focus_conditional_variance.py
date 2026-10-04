import json

import numpy as np
import pytest
from scipy.optimize import check_grad

from research.focus_conditional_motion import fit as mean_fit, predict as mean_predict
from research.focus_conditional_variance import fit, predict, metrics, objective, design_matrix
from research.focus_residual_calibration import VARIANCE_FLOOR


def data():
    rng = np.random.default_rng(734671)
    x = rng.normal(size=(800, 6))
    y = .3 * x[:, :3] + rng.normal(size=(800, 3)) * np.exp(.5 * x[:, :1])
    return x, y


def test_analytic_gradient():
    x, y = data()
    model = mean_fit(x, y)
    design = design_matrix(model, x[:30])
    squared = (y[:30] - mean_predict(model, x[:30])) ** 2
    theta = np.zeros(21)
    error = check_grad(lambda p: objective(p, design, squared)[0], lambda p: objective(p, design, squared)[1], theta)
    assert error < 2e-4


def test_role_guard_before_data_access():
    with pytest.raises(ValueError, match='Only fitting'):
        fit({}, None, None, 'diagnostic')


def test_portable_positive_prediction_and_unchanged_mean():
    x, y = data()
    mean_model = mean_fit(x, y)
    original = json.dumps(mean_model)
    model = fit(mean_model, x, y, 'fitting')
    values = predict(model, x)
    assert values.shape == (800, 3) and (values >= VARIANCE_FLOOR).all()
    assert json.dumps(mean_model) == original
    np.testing.assert_array_equal(values, predict(json.loads(json.dumps(model)), x))
    assert metrics(y, mean_predict(mean_model, x), values)['squared_error_sum'] == float(((y - mean_predict(mean_model, x)) ** 2).sum())
    assert np.corrcoef(x[:, 0], np.log(values[:, 1]))[0, 1] > .8


def test_corrupt_fitted_contract_rejected():
    x, y = data()
    model = fit(mean_fit(x, y), x, y, 'fitting')
    model['coefficients'][0][0] = float('nan')
    with pytest.raises(ValueError):
        predict(model, x)
