from pathlib import Path
import runpy

import pytest

from research.focus_pair_appearance import projector
from research.focus_pair_appearance_head import fit

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = runpy.run_path(str(ROOT / 'tests/test_focus_pair_appearance.py'))['fixture']
CHECK = runpy.run_path(str(ROOT / 'scripts/verify-focus-pair-appearance.py'))['check_smoke']


@pytest.mark.parametrize('arm', ['full', 'lda'])
def test_independent_group_loss_on_saved_smoke_model(arm):
    _, base, feature, stats = FIXTURE()
    model = fit([(base, feature)], projector(stats, 'fitting'), arm, 'fitting')
    actual = CHECK(base, feature, model)
    assert actual['objective'] == pytest.approx(actual['independent_group_objective'], abs=1e-9)
    model['objective'] += .01
    with pytest.raises(ValueError, match='objective'):
        CHECK(base, feature, model)


def test_verifier_rejects_wrong_smoke_weight():
    _, base, feature, stats = FIXTURE()
    model = fit([(base, feature)], projector(stats, 'fitting'), 'lda', 'fitting')
    model['absent_weight'] = 1.
    with pytest.raises(ValueError, match='fixed class weight'):
        CHECK(base, feature, model)
