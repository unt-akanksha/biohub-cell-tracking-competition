"""Verify paired detector training; independent movie scores remain necessary."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from owned_detector_logit_targets import contract
INITIAL_SHA='76f7da6e32c901e3e3f9d2ab870b8a22235b0b91086cfcf909aa41674896a144'
PROBE_SHA='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba'


def verify_arm(result,arm,split,probe):
    identity=result['identity']; train=split['folds'][0]['train']
    expected_loss='Original sparse detector BCE control' if arm=='sparse' else 'Owned frozen view-consensus positive/unlabeled detector BCE'
    if (result['status']!='completed_owned_detector_fit_not_selection' or result['steps']!=1000
        or identity['run_id']!='owned-detector-fit-pair-v1' or identity['max_steps']!=1000
        or identity['initialization_sha256']!=INITIAL_SHA or identity['owned_detector_pu']!=contract()
        or identity['fine_tuning_profile']!='full' or identity['owned_detector_objective']!=arm
        or identity['loss']!=expected_loss or identity['fine_tuning_stems']!=train or len(train)!=120
        or identity['training_stems']!=train or identity['parent_identity']['training_stems']!=train
        or set(train)&set(split['folds'][0]['selection']+split['folds'][0]['audit_order'])
        or identity['teacher_peak_order']!='raw_logits' or identity['teacher_probability_precision']!='float32_before_sigmoid'
        or identity['frozen_modules']!=['transformer','flow','batchnorm_statistics']
        or any(identity[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))
        or result['authorized_for_submission'] is not False
        or any(result[k] is not True for k in ('teacher_unchanged','linker_unchanged','bn_unchanged','detector_changed','strict_reload_identical','probe_replay_passed'))):
        raise ValueError('Complete frozen-component source-only paired detector fit required')
    hashes=result['input_hashes']+result['target_hashes']+[result['checkpoint_sha256']]
    if (len(result['input_hashes'])!=2000 or len(result['target_hashes'])!=4000
        or any(len(h)!=64 or set(h)-set('0123456789abcdef') for h in hashes)
        or result['input_hashes'][:20]!=probe['input_hashes'] or result['target_hashes'][:40]!=probe['target_hashes']
        or [r['step'] for r in result['history']]!=list(range(1,1001))):
        raise ValueError('Exact full history and verified raw-target prefix replay required')
    for row in result['history']:
        if (any(not math.isfinite(row[k]) or row[k]<0 for k in ('loss','grad_norm','elapsed_seconds'))
            or any(row[k]<=0 for k in ('annotations','positive_voxels','unknown_voxels'))
            or not 0<=row['fp32_unit_probability_voxels']<=row['legacy_unit_probability_voxels']):
            raise ValueError('Invalid training numerical/target history')
    for key,step in [('first10_gate',10),('candidate',1000),('control',1000)]:
        gate=result[key]; receipt=gate['pre_motion_patch_receipt']
        if (gate['status']!='passed' or gate['checkpoint_step']!=step or gate['movie']!=train[0] or gate['frames']!=3
            or gate['strict_reload'] is not True or gate['geff_round_trip'] is not True
            or gate['standalone_image_flow'] is not True or gate['predicted_nodes']<=0
            or receipt['views']!=8 or receipt['encode_calls']!=2
            or gate['encode_patch_receipt']!=dict(maximum_nodes_per_frame=2048,action='abort_without_truncation')):
            raise ValueError('Pre-extension and final protected graph gates required')
    if result['candidate']['checkpoint_sha256']!=result['checkpoint_sha256'] or result['control']['checkpoint_sha256']!=INITIAL_SHA:
        raise ValueError('Graph/checkpoint identity mismatch')
    return dict(arm=arm,checkpoint_sha256=result['checkpoint_sha256'],steps=1000,
        optimization_seconds=result['history'][-1]['elapsed_seconds'],last_loss=result['history'][-1]['loss'],
        observed_annotations=sum(r['annotations'] for r in result['history']),
        generated_consensus=sum(r['consensus'] for r in result['history']),
        strict_reload_identical=True,first10_gate_passed=True,frozen_components_unchanged=True,
        final_training_frame_smoke=result['candidate'])


def verify(pair,terminal,results,split,probe):
    if (pair['status']!='completed_detector_pair_not_selection' or pair['steps_per_arm']!=1000
        or pair['paired_inputs_identical'] is not True or pair['paired_targets_identical'] is not True
        or any(pair[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))
        or terminal['status']!='completed' or terminal['run_id']!='owned-detector-fit-pair-v1'
        or not 0<terminal['elapsed_seconds']<=3600 or terminal['submission_performed'] is not False
        or set(results)!={'sparse','pu'}):
        raise ValueError('Completed sequential detector-pair terminal required')
    arms={arm:verify_arm(result,arm,split,probe) for arm,result in results.items()}
    if (pair['checkpoints']!={arm:r['checkpoint_sha256'] for arm,r in results.items()}
        or results['sparse']['input_hashes']!=results['pu']['input_hashes']
        or results['sparse']['target_hashes']!=results['pu']['target_hashes']):
        raise ValueError('Exact paired source inputs, teacher targets and checkpoint identities required')
    return dict(status='verified_detector_fit_pair_not_selection',arms=arms,launcher_seconds=terminal['elapsed_seconds'],
        paired_inputs_identical=True,paired_targets_identical=True,authorized_for_submission=False,
        caveat='Training loss and three-frame checks are not independent tracking evidence; compare complete source-selection movies before further audit')


if __name__=='__main__':
    cache=ROOT/'.biohub/cache/kernel-outputs/owned-detector-fit-pair-v1/owned_detector_fit_pair'
    paths=dict(pair=cache/'outputs/pair_result.json',terminal=cache/'launcher_terminal.json',
        sparse=cache/'outputs/sparse/result.json',pu=cache/'outputs/pu/result.json')
    probe_path=ROOT/'.biohub/cache/kernel-outputs/owned-detector-logit-probe-v1/owned_detector_logit_probe/outputs/result.json'
    if hashlib.sha256(probe_path.read_bytes()).hexdigest()!=PROBE_SHA:
        raise ValueError('Original raw-target probe receipt changed')
    data={k:json.loads(p.read_text()) for k,p in paths.items()}
    report=verify(data['pair'],data['terminal'],{k:data[k] for k in ('sparse','pu')},
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()),json.loads(probe_path.read_text()))
    notebook=ROOT/'kaggle/biohub-owned-detector-fit-pair-v1/biohub-owned-detector-fit-pair-v1.ipynb'
    source=''.join(json.loads(notebook.read_text())['cells'][1]['source'])
    for name,filename in [('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')]:
        assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        expected={k:hashlib.sha256(v.encode()).hexdigest() for k,v in ast.literal_eval(assignment.value).items()}
        if json.loads((cache/filename).read_text())!=expected: raise ValueError('Immutable fit-pair source mismatch')
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    report['notebook_sha256']=hashlib.sha256(notebook.read_bytes()).hexdigest()
    (ROOT/'reports/experiments/owned-detector-fit-pair-v1-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
