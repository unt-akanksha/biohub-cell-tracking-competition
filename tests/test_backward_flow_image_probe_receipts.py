import copy
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/summarize-backward-flow-image-probe.py'))


def fixture():
    control,terminal,split=runpy.run_path(str(ROOT/'tests/test_backward_flow_probe_receipts.py'))['fixture']()
    control['checkpoint_sha256']='c1203ffb03ac98b4aa5b332ec26b4469ba5b9e784e6e095f03a41c533bba5884'
    candidate=copy.deepcopy(control); candidate['checkpoint_sha256']='e'*64
    candidate['identity'].update(run_id='backward-flow-image-probe-v1',loss_profile=dict(image_weight=.25,smoothness_weight=.01))
    for row in candidate['history']:
        sparse=row.pop('loss_um')
        row.update(sparse_loss_um=sparse,image_loss=.1,ssim_loss=.08,boundary_loss=.02,smoothness=.2,
            total_objective=sparse+.025+.002,texture_patches=100,invalid_texture_patches=2)
    candidate_terminal=dict(terminal,run_id='backward-flow-image-probe-v1')
    return candidate,candidate_terminal,split,control,terminal


def test_equal_error_does_not_promote_extra_objective():
    report=M['compare'](*fixture())
    assert report['decision']=='no_paired_probe_gain' and not report['authorized_for_submission']


@pytest.mark.parametrize('fault',['inputs','objective','support','nan','profile'])
def test_nonpaired_or_invalid_image_objective_rejected(fault):
    candidate,terminal,split,control,control_terminal=fixture()
    if fault=='inputs': candidate['input_hashes'][0]='f'*64
    if fault=='objective': candidate['history'][0]['total_objective']=0.
    if fault=='support': candidate['history'][0]['invalid_texture_patches']=101
    if fault=='nan': candidate['history'][0]['image_loss']=float('nan')
    if fault=='profile': candidate['identity']['loss_profile']['image_weight']=1.
    with pytest.raises(ValueError): M['compare'](candidate,terminal,split,control,control_terminal)
