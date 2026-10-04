import copy
from pathlib import Path
import runpy

import numpy as np
import pytest

from research.focus_pair_tree import SETTINGS, VERSION, ROUNDS, derivatives

VERIFY = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/verify-focus-pair-tree-lomo.py'))


def test_independent_group_losses_and_real_null_ties():
    block = dict(offset=np.zeros(5), starts=np.array([0, 3]), sizes=np.array([3, 2]),
                 chosen=np.array([0, 4]), present=np.array([1, 0]))
    scores = np.array([1., 1., 1., -3., -1.])
    actual = VERIFY['independent_group_metrics'](scores, block)
    assert actual['correct_parent'] == actual['correct_absent'] == 1
    expected = np.log(3)+np.log1p(np.exp(-2))
    assert actual['loss_sum'] == pytest.approx(expected)
    weighted = VERIFY['weighted_loss'](scores, block, 3.)
    args = [block[k].astype(np.int64) for k in ('starts', 'sizes', 'chosen', 'present')]
    assert weighted == pytest.approx(derivatives(scores, *args, 3.)[0])
    with pytest.raises(ValueError):
        VERIFY['independent_group_metrics'](scores[:-1], block)


def model():
    return dict(settings=copy.deepcopy(SETTINGS), version=VERSION, rounds=ROUNDS, role='fitting',
        external_baseline_required=True, authorized_for_submission=False, training_groups=2,
        training_choices=5, initial_loss=3., final_loss=2., loss_trace=[3.]*100,
        trees=[dict(nodeid=0, leaf=0.) for _ in range(100)])


@pytest.mark.parametrize('key,value', [('training_choices', 4), ('rounds', 75),
    ('external_baseline_required', False), ('authorized_for_submission', True), ('loss_trace', [3.]*75)])
def test_missing_or_misrepresented_contract_rejected(key, value):
    value_model = model()
    VERIFY['require_tree'](value_model, dict(groups=2, choices=5), False)
    value_model[key] = value
    with pytest.raises(ValueError):
        VERIFY['require_tree'](value_model, dict(groups=2, choices=5), False)


def test_recovered_trace_must_identify_missing_first75_round_losses():
    value_model = model()
    value_model.update(recovered_prefix_rounds=75, loss_trace_start_round=76, loss_trace=[3.]*25)
    VERIFY['require_tree'](value_model, dict(groups=2, choices=5), True)
    value_model['loss_trace_start_round'] = 1
    with pytest.raises(ValueError):
        VERIFY['require_tree'](value_model, dict(groups=2, choices=5), True)
