"""Verify three-arm real GPU evidence before any full-movie motion test."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN='backward-flow-spatial-tta-probe-v2'
PARENT='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
SPLIT='12eca8b1f77b549cebb241bd81ced8f3b4b38bef16d18dce2a551c40d31e9d13'


def verify_result(result,split):
    if (result['status']!='passed_flow_spatial_tta_functionality' or result['checkpoint_sha256']!=PARENT
        or result['split_sha256']!=SPLIT or result['movie']!=split['folds'][0]['train'][0]
        or result['frames']!=3 or result['detections_identical'] is not True
        or result['zero_weight_graph_identical'] is not True or len(result['coordinate_sha256'])!=64
        or len(result['frozen_flow_hashes'])!=3
        or any(r['before']!=r['after'] or len(r['before'])!=64 for r in result['frozen_flow_hashes'])
        or any(result[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Successful frozen three-arm training-only probe required')
    for arm in ('control','optimized','candidate'):
        row=result[arm]; execution=row['motion_execution_receipt']
        if (row['status']!='passed' or row['checkpoint_sha256']!=PARENT or row['frames']!=3
            or row['predicted_nodes']<=0 or row['predicted_nodes']!=result['control']['predicted_nodes']
            or row['strict_reload'] is not True or row['geff_round_trip'] is not True
            or row['standalone_image_flow'] is not True
            or row['encode_patch_receipt']['limit']!=2048
            or row['encode_patch_receipt']['maximum_nodes']>2048
            or row['encode_patch_receipt']['truncation'] is not False
            or row['pre_motion_patch_receipt']['views']!=8
            or row['pre_motion_patch_receipt']['encode_calls']!=2
            or execution['skip_zero_neural'] is not (arm!='control')
            or execution['neural_forward_calls']!=(2 if arm=='control' else 0)
            or execution['zero_weight_skips']!=(0 if arm=='control' else 2)):
            raise ValueError('Real executed policy and bounded identical detections required')
    if result['control']['scorer_counts']!=result['optimized']['scorer_counts']:
        raise ValueError('Runtime shortcut altered official counts')
    flow=result['candidate']['flow_patch_receipt']
    if (flow['version']!=1 or flow['views']!=8 or flow['forward_calls']!=2
        or not 0<flow['maximum_mean_absolute_flow_delta_um']<float('inf')
        or flow['output_precision']!='FP32 arithmetic mean'):
        raise ValueError('Nontrivial finite real vector average required')


def main():
    notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    nb=json.loads(notebook.read_text())
    folder=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/backward_flow_spatial_tta_probe'
    source=''.join(nb['cells'][1]['source'])
    for key,file in (('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==key)
        bundle=ast.literal_eval(node.value)
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in bundle.items()}
        if json.loads((folder/file).read_text())!=expected: raise ValueError('Executed source bundle changed')
    terminal=json.loads((folder/'launcher_terminal.json').read_text())
    if terminal['status']!='completed' or terminal['run_id']!=RUN or not 0<terminal['elapsed_seconds']<=3600:
        raise ValueError('Completed bounded GPU probe required')
    split_path=ROOT/'research/independent_real_baseline_v1_split.json'
    if hashlib.sha256(split_path.read_bytes()).hexdigest()!=SPLIT: raise ValueError('Split changed')
    result_path=folder/'outputs/result.json'
    result=json.loads(result_path.read_text()); verify_result(result,json.loads(split_path.read_text()))
    report=dict(status='verified_flow_spatial_tta_probe_not_selection',result=result,terminal=terminal,
        source_sha256=dict(notebook=hashlib.sha256(notebook.read_bytes()).hexdigest(),
                           result=hashlib.sha256(result_path.read_bytes()).hexdigest()),
        authorized_for_submission=False,
        caveat='Three training frames establish functionality only; cold/cache timing differences do not establish a speedup factor. No full-movie gain or transfer evidence yet.')
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists(): raise ValueError('Refuse to overwrite verified GPU receipt')
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
