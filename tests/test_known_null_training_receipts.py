import copy
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT/'scripts/summarize-known-null-training.py'))


def receipt():
    identity = dict(initialization_sha256=MODULE['INITIAL'],max_steps=100,
        training_stems=['train'],training_profile='full',
        loss='sparse_parent_with_annotated_missing_parent_null_v1',
        known_null=dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False),
        frozen_modules=['unet','detect_head'],selection_opened=False,
        target_audit_opened=False,authorized_for_submission=False)
    smoke = dict(status='passed',checkpoint_step=0,strict_reload=True,geff_round_trip=True,
                 predicted_nodes=292,checkpoint_sha256='a'*64)
    result = dict(status='completed',identity=identity,detector_unchanged=True,
        initial_detector_sha256='d'*64,frozen_detector_sha256='d'*64,
        authorized_for_submission=False,known_null_supervised_total=10,
        sample_hashes=['f'*64]*200,checkpoint_sha256='a'*64,
        before=smoke,after=dict(smoke,checkpoint_step=100),
        history=[dict(step=s,parent_loss=.4,max_nodes=400,known_null_columns=1,
            confident_null_columns=0,positive_links=10,elapsed_seconds=s/2) for s in range(10,101,10)])
    terminal = dict(status='completed',elapsed_seconds=100,submission_performed=False)
    split = dict(folds=[dict(train=['train'])])
    return result, terminal, split


def test_valid_receipts_do_not_authorize_selection_or_submission():
    report = MODULE['verify'](*receipt(),100)
    assert report['known_null_columns'] == 10
    assert not report['authorized_for_submission']


@pytest.mark.parametrize('mutation',['detector','history','null_count','checkpoint','input_coverage','scope'])
def test_inconsistent_training_evidence_is_rejected(mutation):
    result, terminal, split = receipt()
    if mutation == 'detector':
        result['detector_unchanged'] = False
    elif mutation == 'history':
        result['history'].pop()
    elif mutation == 'null_count':
        result['known_null_supervised_total'] = 0
    elif mutation == 'checkpoint':
        result['after']['checkpoint_sha256'] = 'b'*64
    elif mutation == 'input_coverage':
        result['sample_hashes'].pop()
    else:
        result['identity']['target_audit_opened'] = True
    with pytest.raises(ValueError):
        MODULE['verify'](result,terminal,split,100)


def test_changed_probe_inputs_are_rejected():
    result, terminal, split = receipt()
    probe = copy.deepcopy(result)
    probe['sample_hashes'][0] = '0'*64
    with pytest.raises(ValueError):
        MODULE['verify'](result,terminal,split,100,probe)
