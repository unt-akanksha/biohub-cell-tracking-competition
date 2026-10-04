import numpy as np
import pytest
from research.focus_correlated_motion import covariance, metrics, gate
from research.focus_conditional_motion import metrics as diagonal_metrics
from research.focus_residual_calibration import VARIANCE_FLOOR


def test_diagonal_density_replay():
    rng = np.random.default_rng(35)
    y = rng.normal(size=(140, 3)); mean = rng.normal(size=(140, 3)); variance = np.array([2., 3., 4.])
    expected = diagonal_metrics(y, mean, variance)
    actual = metrics(y, mean, np.diag(variance))
    assert actual['squared_error_sum'] == expected['squared_error_sum']
    assert actual['nll_sum'] == pytest.approx(expected['nll_sum'], abs=1e-12)


def test_physical_floor_and_oblique_covariance():
    rng = np.random.default_rng(22)
    errors = rng.normal(size=(1000, 3)) @ np.array([[2., 1., .5], [0., .2, .1], [0., 0., .001]])
    actual = covariance(errors)
    assert np.linalg.eigvalsh(actual - np.diag(VARIANCE_FLOOR)).min() >= -1e-12
    assert abs(actual[0, 1]) > 1
    assert np.isfinite(metrics(errors, np.zeros(3), actual)['nll'])


def test_zero_residuals_have_physical_floor():
    np.testing.assert_allclose(covariance(np.zeros((100, 3))), np.diag(VARIANCE_FLOOR), atol=1e-15)


@pytest.mark.parametrize('bad', [np.eye(2), np.diag([1., -1., 1.]), np.array([[1., 2., 0.], [0., 1., 0.], [0., 0., 1.]])])
def test_reject_invalid_covariance(bad):
    with pytest.raises(ValueError): metrics(np.zeros((100, 3)), np.zeros(3), bad)


def test_gate_requires_all_folds_and_unchanged_errors():
    base = dict(observations=100, nll_sum=200., nll=2., squared_error_sum=100.)
    rows = [dict(stem=str(i), baseline=dict(base), candidate=dict(base, nll_sum=190., nll=1.9)) for i in range(14)]
    assert gate(rows)['passed']
    rows[0]['candidate']['squared_error_sum'] = 99.
    assert not gate(rows)['passed']
    with pytest.raises(ValueError): gate(rows[:13])
