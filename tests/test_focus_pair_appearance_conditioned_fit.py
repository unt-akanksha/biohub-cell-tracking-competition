import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_pair_appearance import projector
from research.focus_pair_appearance_head import fit as original_fit
from research.focus_pair_appearance_conditioned_fit import fit

FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_focus_pair_appearance.py')))['fixture']


def test_original_objective_and_portable_theta(tmp_path):
    _, base, app, stats = FIXTURE()
    projection = projector(stats, 'fitting')
    original = original_fit([(base, app)], projection, 'full', 'fitting')
    model, execution = fit([(base, app)], projection, tmp_path / 'fit', 'fitting')
    assert model['objective'] == pytest.approx(original['objective'], abs=1e-5)
    assert model['counts'] == original['counts'] and model['projection'] == projection
    assert np.all(np.asarray(model['theta'])[4:7] <= .5)
    state = json.loads((tmp_path / 'fit/optimizer-terminal.json').read_text())
    assert state['success'] and state['theta'] == model['theta']
    with np.load(tmp_path / 'fit/coordinates.npz', allow_pickle=False) as saved:
        np.testing.assert_allclose(saved['transform'] @ state['beta'], model['theta'], atol=1e-12)
    assert execution['hessian_relative_error'] < 2e-6


def test_role_rejected_before_access(tmp_path):
    with pytest.raises(ValueError, match='Only fitting'):
        fit(None, None, tmp_path / 'bad', 'diagnostic')
