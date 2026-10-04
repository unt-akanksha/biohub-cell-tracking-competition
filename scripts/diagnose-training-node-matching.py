"""Bounded CPU label-ambiguity screen on already verified training frames."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.training_match_diagnostic import compare_matches


def main():
    report_path=ROOT/'reports/experiments/detector-calibration-full-v1-result.json'
    if hashlib.sha256(report_path.read_bytes()).hexdigest()!='64b7d1c5aa2484111636e55e959d01f54edeb12490ae42a744e8af3d1d94e8f4':
        raise ValueError('Verified full training collection receipt changed')
    collection=json.loads(report_path.read_text())['collection']
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]
    records=collection['records']
    if (len(records)!=360 or set(r['stem'] for r in records)!=set(split['train'])
        or collection['selection_opened'] is not False or collection['target_audit_opened'] is not False):
        raise ValueError('Exact previously opened 120-training-movie scope required')
    folder=ROOT/'.biohub/cache/kernel-outputs/owned-detector-calibration-full-v1/owned_detector_calibration_full/outputs'
    rows=[]
    for row in records:
        if row['artifact']!=f"{row['stem']}_{row['t']:03d}.npz": raise ValueError('Unexpected artifact path')
        path=folder/row['artifact']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: raise ValueError('Frame artifact changed')
        with np.load(path,allow_pickle=False) as payload:
            result=compare_matches(payload['parent_coords'][:,1:]*[1.625,.40625,.40625],
                payload['truth_coords'][:,1:]*[1.625,.40625,.40625])
        rows.append(dict(stem=row['stem'],t=row['t'],**result))
    totals={k:sum(r[k] for r in rows) for k in result}
    output=dict(status='completed_training_match_ambiguity_diagnostic',totals=totals,
        frames_with_more_matches=sum(r['additional_matched']>0 for r in rows),per_frame=rows,
        radius_um=5.,source_collection_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        selection_opened=False,target_audit_opened=False,authorized_for_submission=False,
        caveat='Cached parent FP32 D4 training-frame detections, not exact native-AMP training replay. Stable CPU tie order may differ from Torch. More matched annotations is not proof of better tracking. No metric or graph altered; failed calibration remains rejected.')
    target=ROOT/'reports/experiments/training-node-matching-diagnostic-v1.json'
    if target.exists(): raise ValueError('Refuse to overwrite completed diagnostic')
    target.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='per_frame'},indent=2))


if __name__=='__main__': main()
