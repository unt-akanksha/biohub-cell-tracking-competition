import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'scripts/verify-backward-flow-spatial-tta-probe.py'))


def fixture():
    source=json.loads((ROOT/'reports/experiments/owned-detector-ensemble-probe-v1-result.json').read_text())['result']
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    result=dict(status='passed_flow_spatial_tta_functionality',checkpoint_sha256=M['PARENT'],split_sha256=M['SPLIT'],
        movie=split['folds'][0]['train'][0],frames=3,detections_identical=True,zero_weight_graph_identical=True,
        coordinate_sha256='a'*64,frozen_flow_hashes=[dict(before='f'*64,after='f'*64) for _ in range(3)],
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False)
    for arm in ('control','optimized','candidate'):
        row=copy.deepcopy(source['control'])
        row['motion_execution_receipt']=dict(skip_zero_neural=arm!='control',
            neural_forward_calls=2 if arm=='control' else 0,zero_weight_skips=0 if arm=='control' else 2)
        result[arm]=row
    result['candidate']['flow_patch_receipt']=dict(version=1,views=8,forward_calls=2,
        maximum_mean_absolute_flow_delta_um=.1,output_precision='FP32 arithmetic mean')
    return result,split


def test_complete_three_arm_receipt(): M['verify_result'](*fixture())


@pytest.mark.parametrize('fault',['target','graph','weights','calls','zero','nan','nodes','partial'])
def test_reject_unexecuted_or_changed_policy(fault):
    result,split=fixture()
    if fault=='target': result['target_audit_opened']=True
    if fault=='graph': result['optimized']['scorer_counts']['edge_tp']+=1
    if fault=='weights': result['frozen_flow_hashes'][0]['after']='b'*64
    if fault=='calls': result['optimized']['motion_execution_receipt']['neural_forward_calls']=2
    if fault=='zero': result['candidate']['flow_patch_receipt']['maximum_mean_absolute_flow_delta_um']=0.
    if fault=='nan': result['candidate']['flow_patch_receipt']['maximum_mean_absolute_flow_delta_um']=float('nan')
    if fault=='nodes': result['candidate']['encode_patch_receipt']['maximum_nodes']=2049
    if fault=='partial': result['candidate']['flow_patch_receipt']['forward_calls']=1
    with pytest.raises(ValueError): M['verify_result'](result,split)
