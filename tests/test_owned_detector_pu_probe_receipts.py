import copy
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/summarize-owned-detector-pu-probe.py'))


def fixture():
    cache=ROOT/'.biohub/cache/kernel-outputs/owned-detector-pu-probe-v1/owned_detector_pu_probe'
    return (json.loads((cache/'outputs/result.json').read_text()),
        json.loads((cache/'launcher_terminal.json').read_text()),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()))


def test_original_functionality_receipt_is_not_accuracy_authorization():
    report=M['verify'](*fixture())
    assert report['steps']==10 and not report['authorized_for_submission']


@pytest.mark.parametrize('fault',['scope','teacher','inputs','steps','graph'])
def test_changed_or_incomplete_detector_receipts_fail_closed(fault):
    result,terminal,split=fixture()
    if fault=='scope': result['identity']['fine_tuning_stems'].append(split['folds'][0]['audit_order'][0])
    if fault=='teacher': result['teacher_unchanged']=False
    if fault=='inputs': result['input_hashes'].pop()
    if fault=='steps': result['history'].pop()
    if fault=='graph': result['candidate']['geff_round_trip']=False
    with pytest.raises(ValueError): M['verify'](result,terminal,split)


def test_fp32_profile_cannot_reuse_old_numerical_receipts():
    result,terminal,split=fixture()
    run_id='owned-detector-pu-fp32-probe-v1'
    result['identity']['run_id']=terminal['run_id']=run_id
    with pytest.raises(ValueError): M['verify'](result,terminal,split,run_id=run_id)


def test_corrected_builder_preserves_original_notebook():
    original=ROOT/'kaggle/biohub-owned-detector-pu-probe-v1/biohub-owned-detector-pu-probe-v1.ipynb'
    before=original.read_bytes()
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-owned-detector-pu-fp32-probe.py'))['build']()
    assert original.read_bytes()==before
    assert nb['metadata']['codex']['teacher_probability_precision']=='float32_before_sigmoid'
    assert meta['id']=='indarkarhana/biohub-owned-detector-pu-fp32-probe-v1'
    assert not meta['enable_internet']
