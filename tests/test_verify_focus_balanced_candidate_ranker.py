from pathlib import Path
import runpy

import pytest

from research.focus_balanced_candidate_ranker import fit

ROOT = Path(__file__).resolve().parents[1]
DATA = runpy.run_path(str(ROOT / 'tests/test_focus_balanced_candidate_ranker.py'))['data']
VERIFY = runpy.run_path(str(ROOT / 'scripts/verify-focus-balanced-candidate-ranker.py'))['verify_fit']


def test_independent_group_loss_replays_actual_fitted_parameters():
    data = DATA()
    model = fit(data, 'fitting')
    checked = VERIFY(data, model)
    assert checked['independent_group_objective'] == pytest.approx(model['objective'], abs=1e-9)
    assert checked['fitting_parent'] == 2 and checked['fitting_absent'] == 1


@pytest.mark.parametrize('key,value', [('absent_weight', 1.), ('fitting_parent', 3),
                                      ('fitting_absent', 2), ('parent_weight', 2.),
                                      ('role', 'diagnostic')])
def test_rejects_wrong_weight_counts_or_role(key, value):
    data = DATA()
    model = fit(data, 'fitting')
    model[key] = value
    with pytest.raises(ValueError, match='fitting-only'):
        VERIFY(data, model)


def test_rejects_recorded_objective_that_does_not_match_saved_model():
    data = DATA()
    model = fit(data, 'fitting')
    model['objective'] += .01
    with pytest.raises(ValueError, match='objective'):
        VERIFY(data, model)
