import copy
from pathlib import Path
import runpy

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/summarize-independent-joint-extended.py'))


def fixture(steps):
    split = dict(folds=[dict(train=['train'], selection=['selection'], audit_order=['audit'])])
    identity = dict(max_steps=steps, checkpoint_bn_once=True, joint_training=True,
        training_profile='full', initialization_sha256=MODULE['SOURCE_SHA'],
        loss='sparse_parent_classification_with_null_v1', training_stems=['train'],
        selection_opened=False, target_audit_opened=False)
    def gate(step):
        return dict(status='passed', checkpoint_step=step, strict_reload=True,
                    geff_round_trip=True, movie='train', checkpoint_sha256='a'*64)
    result = dict(status='completed', identity=identity, authorized_for_submission=False,
        sample_hashes=['sample'] * (2*steps), initial_module_hashes={'unet':'a','transformer':'b','detect_head':'c'},
        final_module_hashes={'unet':'d','transformer':'e','detect_head':'f'}, before=gate(0), after=gate(steps),
        step100_smoke=gate(100) if steps > 100 else None, checkpoint_sha256='a'*64,
        history=[dict(step=step, batchnorm_update_deltas={str(i):step for i in range(10)},
                      optimizer_steps_min=step, optimizer_steps_max=step) for step in range(10, steps+1, 10)])
    terminal = dict(status='completed', run_id='independent-joint-extended-v1', elapsed_seconds=100,
                    declared_budget_seconds=7200, submission_performed=False)
    return result, terminal, split


@pytest.mark.parametrize('steps', [100,6000])
def test_valid_receipts(steps):
    assert MODULE['validate'](*fixture(steps), steps)['status'] == 'verified_training_receipts'


def test_bad_bn_optimizer_reload_or_frozen_module_rejected():
    result, terminal, split = fixture(6000)
    for mutate in (
        lambda r: r['history'][0]['batchnorm_update_deltas'].update({'0':20}),
        lambda r: r['history'][-1].update(optimizer_steps_min=5300),
        lambda r: r['after'].update(checkpoint_sha256='b'*64),
        lambda r: r['final_module_hashes'].update(unet='a'),
        lambda r: r.update(step100_smoke=dict(r['step100_smoke'], strict_reload=False)),
    ):
        bad = copy.deepcopy(result)
        mutate(bad)
        with pytest.raises(ValueError):
            MODULE['validate'](bad, terminal, split, 6000)
