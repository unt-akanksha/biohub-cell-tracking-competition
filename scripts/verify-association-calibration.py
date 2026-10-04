"""Verify frozen-network probe/full-fit receipts, never authorize submission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
RUN = runpy.run_path(str(ROOT/'scripts/run-association-calibration.py'))


def verify(result,terminal,split,profile,probe=None):
    fitting,diagnostic = RUN['scope'](split,profile)
    label = 'probe' if profile=='probe' else 'fit'
    if (result['status']!='completed_calibration_not_tracking_validation' or result['profile']!=profile
        or terminal['status']!='completed' or terminal['run_id']!=f'association-calibration-{label}-v1'
        or terminal['declared_budget_seconds']!=3600 or not 0<terminal['elapsed_seconds']<=3600
        or terminal['submission_performed'] is not False or not result['functionality_passed']
        or result['checkpoint_sha256']!=RUN['CHECKPOINT_SHA'] or result['split_sha256']!=RUN['SPLIT_SHA']
        or result['fitting_stems']!=fitting or result['diagnostic_stems']!=diagnostic
        or result['frozen_before']!=result['frozen_after']
        or result['frozen_before']['flow']!='e82b7255fb2cded608a800fb6627ec4972043491e636e22a0dcbd66fc5c93779'
        or any(result[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Exact completed frozen calibration receipt required')
    records = result['records']
    expected = [(role,stem) for role,stems in [('fitting',fitting),('diagnostic',diagnostic)] for stem in stems for _ in range(3)]
    if [(r['role'],r['stem']) for r in records]!=expected or len({(r['stem'],r['t_start']) for r in records})!=len(expected):
        raise ValueError('Incomplete or duplicate calibration windows')
    for r in records:
        if (not 0<=r['t_start']<=98 or not 0<=r['supervised_columns']<=r['target_nodes']
            or not 0<=r['known_null_columns']<=r['supervised_columns']):
            raise ValueError('Invalid supervision counts')
        for name in ('cache_sha256','input_sha256'):
            if len(r[name])!=64 or set(r[name])-set('0123456789abcdef'):
                raise ValueError('Missing immutable input/cache hashes')
    fit = result['fit']
    bounds = [[0.,1.],[.25,4.],[-12.,4.]]
    if (fit['bounds']!=bounds or len(fit['parameters'])!=3 or not fit['converged']
        or any(not math.isfinite(x) or not lo<=x<=hi for x,(lo,hi) in zip(fit['parameters'],bounds))
        or fit['fit_metrics']['nll']>fit['flow_control']['nll']+1e-9):
        raise ValueError('Bounded converged calibration required')
    for role,metrics in [('fitting',fit['fit_metrics']),('diagnostic',result['diagnostic']['calibrated'])]:
        if (metrics['columns']!=sum(r['supervised_columns'] for r in records if r['role']==role)
            or metrics['known_null_columns']!=sum(r['known_null_columns'] for r in records if r['role']==role)
            or metrics['columns']<=0 or not math.isfinite(metrics['nll'])):
            raise ValueError('Calibration metric accounting mismatch')
    if profile=='full':
        if probe is None or result['probe_inputs_replayed'] is not True or fit['fit_metrics']['known_null_columns']<=0:
            raise ValueError('Full fit requires replayed probe and real null examples')
        lookup = {(r['stem'],r['t_start']):r for r in records}
        for old in probe['records']:
            now = lookup[(old['stem'],old['t_start'])]
            if any(old[k]!=now[k] for k in ('cache_sha256','input_sha256')):
                raise ValueError('Probe replay mismatch')
    return dict(status='verified_calibration_not_tracking_candidate',profile=profile,
        checkpoint_sha256=result['checkpoint_sha256'],parameters=fit['parameters'],
        windows=len(records),launcher_seconds=terminal['elapsed_seconds'],result=result,
        authorized_for_submission=False)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--profile',choices=('probe','full'),default='probe')
    p.add_argument('--version',type=int,default=1)
    args = p.parse_args()
    label = 'probe' if args.profile=='probe' else 'fit'
    cache = ROOT/f'.biohub/cache/kernel-outputs/association-calibration-{label}-v{args.version}'/f'association_calibration_{label}'
    result = json.loads((cache/'outputs/result.json').read_text())
    terminal = json.loads((cache/'launcher_terminal.json').read_text())
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    probe_path = ROOT/'.biohub/cache/kernel-outputs/association-calibration-probe-v1/association_calibration_probe/outputs/result.json'
    probe = json.loads(probe_path.read_text()) if args.profile=='full' else None
    report = verify(result,terminal,split,args.profile,probe)
    report['source_sha256'] = {k:hashlib.sha256((cache/path).read_bytes()).hexdigest()
        for k,path in [('result','outputs/result.json'),('terminal','launcher_terminal.json')]}
    (ROOT/f'reports/experiments/association-calibration-{label}-v{args.version}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='result'},indent=2))
