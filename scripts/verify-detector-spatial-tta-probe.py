"""Paired native/D4 training-frame functionality receipt, never promotion."""
import hashlib
import json
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
RUN=runpy.run_path(str(ROOT/'scripts/run-detector-spatial-tta-probe.py'))


def verify(result,terminal,split):
    if (result['status']!='passed_detector_tta_functionality' or result['checkpoint_sha256']!=RUN['CHECKPOINT_SHA']
        or result['split_sha256']!=RUN['SPLIT_SHA'] or result['movie']!=split['folds'][0]['train'][0]
        or result['frames']!=3 or terminal['status']!='completed'
        or terminal['run_id']!='detector-spatial-tta-probe-v1' or terminal['declared_budget_seconds']!=3600
        or not 0<terminal['elapsed_seconds']<=3600 or terminal['submission_performed'] is not False
        or result['linking_policy']!=dict(neural_weight=0.,spatial_weight=1.,null_logit=-4.5)
        or any(result[k] is not False for k in ('selection_opened','target_audit_opened','authorized_for_submission'))):
        raise ValueError('Exact completed training-only detector probe required')
    for arm in ('control','candidate'):
        row=result[arm]
        if (row['status']!='passed' or row['device']!='cuda' or row['checkpoint_step']!=1000
            or row['checkpoint_sha256']!=RUN['CHECKPOINT_SHA'] or row['movie']!=result['movie'] or row['frames']!=3
            or not row['strict_reload'] or not row['geff_round_trip'] or not row['standalone_image_flow']
            or row['predicted_nodes']<=0 or row['authorized_for_submission'] is not False):
            raise ValueError('Both paired real-image arms must pass')
    receipt=result['candidate']['pre_motion_patch_receipt']
    if (result['control']['pre_motion_patch_receipt'] is not None or receipt['views']!=8
        or receipt['encode_calls']!=2 or receipt['features']!='native unchanged'
        or receipt['maximum_mean_absolute_logit_delta']<=0):
        raise ValueError('Exactly two image pairs with real detector averaging required')
    return dict(status='verified_detector_tta_probe_not_selection',result=result,
        launcher_seconds=terminal['elapsed_seconds'],authorized_for_submission=False)


if __name__=='__main__':
    cache=ROOT/'.biohub/cache/kernel-outputs/detector-spatial-tta-probe-v1/detector_spatial_tta_probe'
    paths=dict(result=cache/'outputs/result.json',terminal=cache/'launcher_terminal.json')
    report=verify(*(json.loads(paths[k].read_text()) for k in ('result','terminal')),
        json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text()))
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/detector-spatial-tta-probe-v1.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='result'},indent=2))
