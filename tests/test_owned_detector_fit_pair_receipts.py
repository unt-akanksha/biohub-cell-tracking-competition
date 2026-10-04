import copy
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/summarize-owned-detector-fit-pair.py'))


def fixture():
    probe=json.loads((ROOT/'.biohub/cache/kernel-outputs/owned-detector-logit-probe-v1/owned_detector_logit_probe/outputs/result.json').read_text())
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    results={}
    for arm,digest in [('sparse','a'*64),('pu','b'*64)]:
        result=copy.deepcopy(probe)
        result.update(status='completed_owned_detector_fit_not_selection',steps=1000,checkpoint_sha256=digest,probe_replay_passed=True)
        result['identity'].update(run_id='owned-detector-fit-pair-v1',max_steps=1000,fine_tuning_profile='full',owned_detector_objective=arm,
            fine_tuning_stems=split['folds'][0]['train'],loss='Original sparse detector BCE control' if arm=='sparse' else 'Owned frozen view-consensus positive/unlabeled detector BCE')
        result['input_hashes']=probe['input_hashes']+['c'*64]*1980
        result['target_hashes']=probe['target_hashes']+['d'*64]*3960
        result['history']=[dict(probe['history'][0],step=s,elapsed_seconds=float(s)) for s in range(1,1001)]
        result['first10_gate']=copy.deepcopy(result['candidate'])
        result['candidate'].update(checkpoint_step=1000,checkpoint_sha256=digest)
        for key in ('candidate','control','first10_gate'):
            result[key]['encode_patch_receipt']=dict(maximum_nodes_per_frame=2048,action='abort_without_truncation')
        results[arm]=result
    pair=dict(status='completed_detector_pair_not_selection',steps_per_arm=1000,paired_inputs_identical=True,paired_targets_identical=True,
        checkpoints={k:v['checkpoint_sha256'] for k,v in results.items()},selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    terminal=dict(status='completed',run_id='owned-detector-fit-pair-v1',elapsed_seconds=2200.,submission_performed=False)
    return pair,terminal,results,split,probe


def test_verified_pair_never_claims_independent_tracking_gain():
    result=M['verify'](*fixture())
    assert result['paired_inputs_identical'] and not result['authorized_for_submission']
    assert set(result['arms'])=={'sparse','pu'}


@pytest.mark.parametrize('fault',['nonpaired','prefix','partial','scope','gate','freeze','profile','time'])
def test_invalid_fit_pair_rejected(fault):
    pair,terminal,results,split,probe=fixture()
    result=results['pu']
    if fault=='nonpaired': result['input_hashes'][50]='f'*64
    if fault=='prefix': result['target_hashes'][0]='f'*64
    if fault=='partial': result['history'].pop()
    if fault=='scope': result['identity']['fine_tuning_stems']=['44b6_invalid']
    if fault=='gate': result['first10_gate']['strict_reload']=False
    if fault=='freeze': result['teacher_unchanged']=False
    if fault=='profile': result['identity']['owned_detector_objective']='sparse'
    if fault=='time': terminal['elapsed_seconds']=3601
    with pytest.raises(ValueError): M['verify'](pair,terminal,results,split,probe)
