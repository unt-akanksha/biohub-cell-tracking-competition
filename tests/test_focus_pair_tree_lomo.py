from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_pair_appearance_head import fit, metrics
from research.focus_pair_appearance import projector
from research.focus_pair_tree import SETTINGS, VERSION, ROUNDS

ROOT = Path(__file__).resolve().parents[1]
DRIVER = runpy.run_path(str(ROOT/'scripts/fit-focus-pair-tree-lomo.py'))
FIXTURE = runpy.run_path(str(Path(__file__).with_name('test_focus_pair_appearance.py')))['fixture']


def test_preparation_and_zero_residual_match_exact_linear_control(tmp_path):
    _, base, app, stats = FIXTURE()
    baseline = fit([(base, app)], projector(stats, 'fitting'), 'full', 'fitting')
    x, margin, groups = DRIVER['prepare']([(base, app)], baseline, tmp_path)
    assert x.shape == (12, 72) and x.dtype == np.float32
    assert len(margin) == len(base['offset'])
    np.testing.assert_array_equal(groups['chosen'], base['chosen'])
    zero = dict(settings=SETTINGS, version=VERSION, trees=[dict(nodeid=0, leaf=0.) for _ in range(ROUNDS)])
    assert DRIVER['model_metrics']((base, app), baseline, zero) == metrics([(base, app)], baseline)


def test_no_implicit_final_fit_or_submission():
    source = (ROOT/'scripts/fit-focus-pair-tree-lomo.py').read_text()
    assert 'final_model=None' in source and 'authorized_for_submission=False' in source
    assert 'reused = held ==' in source
    assert "del training, samples, moments" in source
