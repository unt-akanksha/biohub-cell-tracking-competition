"""Verify detector optimization/reload evidence; never claim held-out gains."""
import ast
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from owned_detector_pu import contract
CHECKPOINT_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'


def verify(result,terminal,split,run_id='owned-detector-pu-probe-v1'):
    identity=result['identity']; fold=split['folds'][0]
    target_contract=contract()
    if run_id=='owned-detector-logit-probe-v1':
        from owned_detector_logit_targets import contract as raw_contract
        target_contract=raw_contract()
    if (result['status']!='passed_owned_detector_pu_functionality' or result['steps']!=10
        or terminal['status']!='completed' or terminal['run_id']!=run_id
        or not 0<terminal['elapsed_seconds']<=3600 or terminal['submission_performed'] is not False
        or identity['run_id']!=run_id or identity['max_steps']!=10
        or identity['initialization_sha256']!=CHECKPOINT_SHA or identity['owned_detector_pu']!=target_contract
        or identity['training_stems']!=fold['train'] or identity['parent_identity']['training_stems']!=fold['train']
        or identity['fine_tuning_stems']!=fold['train'][:4]
        or identity['frozen_modules']!=['transformer','flow','batchnorm_statistics']
        or any(identity[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))
        or result['authorized_for_submission'] is not False
        or any(result[k] is not True for k in ('teacher_unchanged','linker_unchanged','bn_unchanged','detector_changed','strict_reload_identical'))):
        raise ValueError('Completed source-only detector probe with frozen components required')
    if run_id not in ('owned-detector-pu-probe-v1','owned-detector-pu-fp32-probe-v1','owned-detector-logit-probe-v1'):
        raise ValueError('Unregistered probe profile')
    if run_id in ('owned-detector-pu-fp32-probe-v1','owned-detector-logit-probe-v1'):
        if (identity.get('teacher_probability_precision')!='float32_before_sigmoid'
            or any(not 0<=r['fp32_unit_probability_voxels']<=r['legacy_unit_probability_voxels'] for r in result['history'])):
            raise ValueError('FP32-before-sigmoid target provenance required')
    if run_id=='owned-detector-logit-probe-v1' and identity.get('teacher_peak_order')!='raw_logits':
        raise ValueError('Raw-logit peak ordering required')
    hashes=result['input_hashes']+result['target_hashes']+[result['checkpoint_sha256']]
    if (len(result['input_hashes'])!=20 or len(result['target_hashes'])!=40
        or any(len(h)!=64 or set(h)-set('0123456789abcdef') for h in hashes)
        or [r['step'] for r in result['history']]!=list(range(1,11))):
        raise ValueError('Exact ten-step input/target history required')
    for row in result['history']:
        if (any(not math.isfinite(row[k]) or row[k]<0 for k in ('loss','grad_norm','elapsed_seconds'))
            or any(row[k]<=0 for k in ('annotations','positive_voxels','unknown_voxels'))
            or any(row[k]<0 for k in ('consensus','boundary_annotations','background_voxels'))):
            raise ValueError('Invalid optimization or sparse target evidence')
    for name,sha in [('control',CHECKPOINT_SHA),('candidate',result['checkpoint_sha256'])]:
        item=result[name]; receipt=item['pre_motion_patch_receipt']
        if (item['status']!='passed' or item['movie']!=fold['train'][0] or item['frames']!=3
            or item['checkpoint_sha256']!=sha or item['strict_reload'] is not True or item['geff_round_trip'] is not True
            or item['standalone_image_flow'] is not True or item['predicted_nodes']<=0
            or receipt['views']!=8 or receipt['encode_calls']!=2 or receipt['maximum_mean_absolute_logit_delta']<=0):
            raise ValueError('Paired D4/flow graph functionality evidence required')
    return dict(status='verified_owned_detector_pu_functionality_not_selection',checkpoint_sha256=result['checkpoint_sha256'],
        steps=10,launcher_seconds=terminal['elapsed_seconds'],optimization_seconds=result['history'][-1]['elapsed_seconds'],
        input_count=20,target_count=40,all_frozen_components_unchanged=True,detector_changed=True,
        control=result['control'],candidate=result['candidate'],authorized_for_submission=False,
        caveat='Ten updates and three training frames only; larger fit and independent complete-movie tracking evidence still required')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--fp32',action='store_true')
    parser.add_argument('--logits',action='store_true')
    args=parser.parse_args()
    run_id='owned-detector-pu-fp32-probe-v1' if args.fp32 else 'owned-detector-pu-probe-v1'
    work='owned_detector_pu_fp32_probe' if args.fp32 else 'owned_detector_pu_probe'
    if args.logits:
        if args.fp32: parser.error('Choose one numerical profile')
        run_id,work='owned-detector-logit-probe-v1','owned_detector_logit_probe'
    cache=ROOT/'.biohub/cache/kernel-outputs'/run_id/work
    paths=dict(result=cache/'outputs/result.json',terminal=cache/'launcher_terminal.json')
    result=json.loads(paths['result'].read_text())
    report=verify(result,json.loads(paths['terminal'].read_text()),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()),run_id=run_id)
    if args.fp32 or args.logits:
        original=ROOT/'.biohub/cache/kernel-outputs/owned-detector-pu-probe-v1/owned_detector_pu_probe/outputs/result.json'
        if hashlib.sha256(original.read_bytes()).hexdigest()!='d412a2282e0b19447140b61aa4b3203521cd0b65bdc30d7151a6bb1cce8779b4':
            raise ValueError('Original numerical probe receipt required')
        baseline=json.loads(original.read_text())
        if result['input_hashes']!=baseline['input_hashes']:
            raise ValueError('Numerical repair must replay the exact same training inputs')
        report['precision_comparison']=dict(input_replay_identical=True,
            legacy_consensus=sum(r['consensus'] for r in baseline['history']),
            candidate_consensus=sum(r['consensus'] for r in result['history']),
            peak_order=result['identity'].get('teacher_peak_order','probabilities'),
            legacy_unit_probability_voxels=sum(r['legacy_unit_probability_voxels'] for r in result['history']),
            fp32_unit_probability_voxels=sum(r['fp32_unit_probability_voxels'] for r in result['history']))
    notebook=ROOT/'kaggle'/('biohub-'+run_id)/('biohub-'+run_id+'.ipynb')
    source=''.join(json.loads(notebook.read_text())['cells'][1]['source'])
    for name,filename in [('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')]:
        assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in ast.literal_eval(assignment.value).items()}
        if json.loads((cache/filename).read_text())!=expected:
            raise ValueError('Immutable detector-probe source mismatch')
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    report['notebook_sha256']=hashlib.sha256(notebook.read_bytes()).hexdigest()
    (ROOT/'reports/experiments'/(run_id+'-result.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
