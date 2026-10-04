import copy

import numpy as np
import pytest

from research.focus_pair_tree import derivatives, validate_groups, fit, portable_predict, SETTINGS, VERSION, ROUNDS


def groups():
    return (np.array([0, 3], np.int64), np.array([3, 2], np.int64),
            np.array([1, 4], np.int64), np.array([1, 0], np.int64))


def test_gradient_shift_and_diagonal_bound():
    score = np.array([.3, -.7, -4.5, 1.2, -4.5])
    a = groups()
    loss, grad, bound = derivatives(score, *a, 3.)
    numerical = np.empty(5)
    curvature = np.empty((5, 5))
    for i in range(5):
        step = np.zeros(5)
        step[i] = 1e-5
        plus = derivatives(score+step, *a, 3.)
        minus = derivatives(score-step, *a, 3.)
        numerical[i] = (plus[0]-minus[0])/2e-5
        curvature[:, i] = (plus[1]-minus[1])/2e-5
    np.testing.assert_allclose(numerical, grad, atol=1e-8)
    assert np.linalg.eigvalsh(np.diag(bound)-curvature).min() > -1e-9
    shifted = derivatives(score+np.array([100., 100., 100., -20., -20.]), *a, 3.)
    assert shifted[0] == pytest.approx(loss, abs=1e-12)
    np.testing.assert_allclose(shifted[1], grad, atol=1e-12)
    assert abs(grad[:3].sum()) < 1e-12 and abs(grad[3:].sum()) < 1e-12


@pytest.mark.parametrize('field', [0, 1, 2, 3])
def test_incomplete_unknown_misaligned_groups_rejected(field):
    a = list(groups())
    a[field] = a[field].copy()
    a[field][0] = -1
    with pytest.raises(ValueError):
        validate_groups(*a, 5)


def test_role_before_import_or_access(tmp_path):
    with pytest.raises(ValueError, match='Only fitting'):
        fit(None, None, None, None, None, None, None, tmp_path/'no', 'diagnostic')
    assert not (tmp_path/'no').exists()


def example_model():
    tree = dict(nodeid=0, split='f7', split_condition=.25, yes=1, no=2, missing=1,
                children=[dict(nodeid=1, leaf=.05), dict(nodeid=2, leaf=-.05)])
    return dict(version=VERSION, settings=SETTINGS, trees=[copy.deepcopy(tree) for _ in range(ROUNDS)])


def test_float32_strict_boundary_and_bounded_leaf():
    x = np.zeros((3, 72))
    x[:, 7] = [np.nextafter(np.float32(.25), np.float32(0)), .25, .25+1e-10]
    result = portable_predict(example_model(), x)
    assert result[0] > 0 and result[1] < 0 and result[1] == result[2]
    model = example_model()
    model['trees'][0]['children'][0]['leaf'] = 2.
    with pytest.raises(ValueError, match='bounded'):
        portable_predict(model, x)


def test_finite_feature_guard():
    with pytest.raises(ValueError):
        portable_predict(example_model(), np.full((1, 72), np.nan))
