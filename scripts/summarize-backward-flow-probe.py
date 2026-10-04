"""Audit a motion-learning probe, never interpret its errors as tracking scores."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(result,terminal,split_bytes,steps=100,run_id='backward-flow-probe-v1'):
    identity = result['identity']
    fold = json.loads(split_bytes)['folds'][0]
    diagnostic = fold['train'][::5]
    fitting = [s for s in fold['train'] if s not in diagnostic]
    if (result['status'] != 'completed_probe_not_tracking_candidate'
        or terminal['status'] != 'completed' or terminal['run_id'] != run_id
        or terminal['declared_budget_seconds'] != 3600 or not 0 < terminal['elapsed_seconds'] <= 3600
        or terminal['submission_performed'] is not False or result['steps'] != steps
        or identity['max_steps'] != steps or identity['seed'] != 20260910
        or identity['fitting_stems'] != fitting or identity['diagnostic_stems'] != diagnostic
        or identity['split_sha256'] != hashlib.sha256(split_bytes).hexdigest()
        or any(identity[k] is not False for k in ('public_checkpoint_loaded','selection_opened','target_audit_opened','authorized_for_submission'))
        or result['authorized_for_submission'] is not False or result['strict_reload_identical'] is not True):
        raise ValueError('Completed bounded independent probe receipts required')
    hashes = result['input_hashes']+[result['checkpoint_sha256']]
    if len(result['input_hashes']) != 2*steps or any(len(s)!=64 or set(s)-set('0123456789abcdef') for s in hashes):
        raise ValueError('Exact input coverage and checkpoint hash required')
    if [r['step'] for r in result['history']] != list(range(1,steps+1)):
        raise ValueError('Missing optimization history')
    profile=identity.get('loss_profile')
    if profile is not None and (profile!=dict(image_weight=.25,smoothness_weight=.01)
        or steps!=100 or run_id!='backward-flow-image-probe-v1' or identity.get('run_id')!=run_id):
        raise ValueError('Only the frozen bounded image-loss probe is allowed')
    loss_key='total_objective' if profile is not None else 'loss_um'
    for row in result['history']:
        if (row['observed_links'] <= 0 or row['boundary_excluded'] < 0
            or any(not math.isfinite(row[k]) or row[k] < 0 for k in (loss_key,'grad_norm','elapsed_seconds'))):
            raise ValueError('Invalid real supervision or numerical training history')
        if profile is not None:
            fields=('sparse_loss_um','image_loss','ssim_loss','boundary_loss','smoothness')
            if (any(not math.isfinite(row[k]) or row[k]<0 for k in fields)
                or row['ssim_loss']>1+1e-6 or row['texture_patches']<=0
                or not 0<=row['invalid_texture_patches']<=row['texture_patches']
                or not math.isclose(row['image_loss'],row['ssim_loss']+row['boundary_loss'],rel_tol=1e-5,abs_tol=1e-6)
                or not math.isclose(row['total_objective'],row['sparse_loss_um']+.25*row['image_loss']+.01*row['smoothness'],rel_tol=1e-5,abs_tol=1e-6)):
                raise ValueError('Invalid image objective terms or support')
    for stage in ('before','after'):
        item = result[stage]
        rows = item['per_movie']
        if [r['stem'] for r in rows] != diagnostic or any(r['n'] <= 0 for r in rows):
            raise ValueError('Exact24 diagnostic windows required')
        for key,value in item['counts'].items():
            total = sum(r[key] for r in rows)
            if not math.isfinite(value) or value < 0 or not math.isclose(value,total,rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Diagnostic counts mismatch')
        for name,key,scale in [('mae_um','absolute',3),('endpoint_um','endpoint',1),
                              ('zero_mae_um','zero_absolute',3),('zero_endpoint_um','zero_endpoint',1),
                              ('median_mae_um','median_absolute',3),('median_endpoint_um','median_endpoint',1)]:
            expected = item['counts'][key]/(scale*item['counts']['n'])
            if not math.isclose(item[name],expected,rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Incorrect pooled motion error')
    before,after = result['before'],result['after']
    for key in ('zero_mae_um','zero_endpoint_um','median_mae_um','median_endpoint_um'):
        if before[key] != after[key]:
            raise ValueError('Frozen diagnostic controls changed')
    if not math.isclose(before['mae_um'],before['zero_mae_um'],rel_tol=0,abs_tol=1e-6):
        raise ValueError('Initialization must be zero motion')
    passed = (after['mae_um'] < min(after['zero_mae_um'],after['median_mae_um'])
              and after['endpoint_um'] < min(after['zero_endpoint_um'],after['median_endpoint_um']))
    if result['small_fit_gate_passed'] is not passed:
        raise ValueError('Motion-learning gate declaration mismatch')
    return dict(status='verified_motion_probe_not_tracking_validation',checkpoint_sha256=result['checkpoint_sha256'],
        parameters=identity['parameters'],fitting_windows=identity['fitting_windows'],steps=steps,
        diagnostic_links=after['counts']['n'],launcher_seconds=terminal['elapsed_seconds'],
        optimization_seconds=result['history'][-1]['elapsed_seconds'],small_fit_gate_passed=passed,
        errors={k:after[k] for k in ('mae_um','endpoint_um','zero_mae_um','zero_endpoint_um','median_mae_um','median_endpoint_um')},
        observed_training_links=sum(r['observed_links'] for r in result['history']),
        authorized_for_submission=False,caveat='24 short training-only diagnostic windows; not complete movies, embryo audit or tracking score')


if __name__ == '__main__':
    cache = ROOT/'.biohub/cache/kernel-outputs/backward-flow-probe-v1/backward_flow_probe'
    paths = dict(result=cache/'outputs/result.json',terminal=cache/'launcher_terminal.json')
    report = verify(*(json.loads(paths[k].read_text()) for k in ('result','terminal')),
                    (ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    notebook = ROOT/'kaggle/biohub-backward-flow-probe-v1/biohub-backward-flow-probe-v1.ipynb'
    source = ''.join(json.loads(notebook.read_text())['cells'][1]['source'])
    for name,filename in [('sources','source_hashes.json'),('runtime_sources','runtime_hashes.json')]:
        assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id == name)
        expected = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in ast.literal_eval(assignment.value).items()}
        if json.loads((cache/filename).read_text()) != expected:
            raise ValueError('Immutable notebook source receipt mismatch')
    report['source_sha256'] = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    report['notebook_sha256'] = hashlib.sha256(notebook.read_bytes()).hexdigest()
    (ROOT/'reports/experiments/backward-flow-probe-v1-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
