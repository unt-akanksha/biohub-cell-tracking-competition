import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_quadratic_presence import expand, fit, predict_offset, screen


def sample():
    rng = np.random.default_rng(93017)
    x = rng.normal(size=(300, 7))
    y = (x[:, 0] * x[:, 1] + rng.normal(size=300) > 0).astype(int)
    return dict(context=x, offset=np.zeros(300), present=y)


def test_quadratic_basis_exact_and_complete():
    x = np.arange(7., dtype=float)[None, :]
    basis = expand(x)
    assert basis.shape == (1, 35)
    np.testing.assert_array_equal(basis[:, :7], x)
    assert basis[0, 7 + 7] == 1.  # First term of row i=1 is x1*x1.
    assert basis[0, -1] == 36.


def test_role_guard_precedes_data_access():
    with pytest.raises(ValueError, match='Only fitting'):
        fit({}, 'diagnostic')


def test_parameter_roundtrip_and_label_free_application():
    data = sample()
    model = fit(data, 'fitting')
    got = predict_offset(dict(context=data['context'], offset=data['offset']), model)
    recovered = json.loads(json.dumps(model, allow_nan=False))
    np.testing.assert_array_equal(got, predict_offset(data, recovered))
    assert ((got >= 0) == data['present']).mean() > .65
    np.testing.assert_allclose(model['mean'], data['context'].mean(0))
    before = json.dumps(model)
    predict_offset(dict(context=data['context'] + 100, offset=data['offset']), model)
    assert json.dumps(model) == before


def test_bad_inputs_rejected():
    data = sample()
    data['context'][0, 0] = np.nan
    with pytest.raises(ValueError):
        fit(data, 'fitting')


def test_screen_cannot_pass_nll_only_gain():
    original = dict(loss_sum=10., known_parent=10, known_absent=3, correct_parent=9, correct_absent=2, nll=10/13)
    candidate = dict(original, loss_sum=9., nll=9/13, correct_absent=1)
    folds = [dict(held_out=str(i), original=original, linear=original, quadratic=candidate) for i in range(12)]
    gate = screen(folds)
    assert not gate['passed'] and not gate['gates']['absent_not_lower']


def test_actual_loader_opens_only_twelve_fitting_summary_arrays(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    allowed = {'6bba_' + value for value in (
        '6479435d', 'df673a83', '767a1e17', '2312ac41', '971fa5e0', 'cf35214c',
        'cff5865f', 'edf14583', '1ebfb80d', '7d3058ae', '4f99ce20', '57b7cc1e')}
    original_load = np.load
    opened = []

    def only_fitting(path, *args, **kwargs):
        assert Path(path).stem in allowed
        opened.append(Path(path).stem)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', only_fitting)
    runner = runpy.run_path(str(root / 'scripts/fit-focus-quadratic-presence.py'))
    movies, evidence = runner['load_fitting']()
    assert len(opened) == 12 and set(opened) == set(movies) == allowed
    assert sum(len(d['present']) for d in movies.values()) == 10915
    assert evidence['diagnostic_arrays_opened'] is False
