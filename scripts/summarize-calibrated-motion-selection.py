"""Require gains over strongest flow and unchanged-weight neural control."""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT/'scripts/summarize-image-motion-linker-selection.py'))


def compare(candidate,manifest,training,native,causal,parent,flow,uncalibrated,calibration):
    receipt = manifest.get('association_calibration',{})
    if (calibration['status']!='verified_calibration_not_tracking_candidate' or calibration['profile']!='full'
        or receipt.get('parameters')!=calibration['parameters']
        or receipt.get('result_sha256')!=calibration['source_sha256']['result']
        or receipt.get('checkpoint_sha256')!=candidate['checkpoint_sha256']
        or calibration['checkpoint_sha256']!=candidate['checkpoint_sha256']
        or receipt.get('frozen_hashes')!=calibration['result']['frozen_after']
        or uncalibrated['checkpoint_sha256']!=candidate['checkpoint_sha256']):
        raise ValueError('Exact calibration and unchanged neural checkpoint required')
    report = BASE['compare'](candidate,manifest,training,native,causal,parent,flow)
    if [r['stem'] for r in uncalibrated['per_movie']]!=[r['stem'] for r in candidate['per_movie']]:
        raise ValueError('Complete uncalibrated control required')
    for row,previous in zip(candidate['per_movie'],uncalibrated['per_movie']):
        if any(row[k]!=previous[k] for k in ('num_pred_nodes','node_recall','total_node_ratio')):
            raise ValueError('Calibration changed reference detections')
    report.update(status='verified_calibrated_motion_selection',calibration=receipt,
        delta_vs_uncalibrated=candidate['summary']['score']-uncalibrated['summary']['score'],
        regressions_vs_uncalibrated=[r['stem'] for r,p in zip(candidate['per_movie'],uncalibrated['per_movie'])
            if r['adj_edge_jaccard']<p['adj_edge_jaccard']-1e-12],
        caveat='Frozen networks; coefficients fitted only within original training pool; source selection only')
    if report['delta_vs_uncalibrated']<=0:
        report['decision']='not_strongest_standalone'
    return report


if __name__=='__main__':
    paths=dict(candidate=ROOT/'.biohub/cache/kernel-outputs/calibrated-motion-scoring-v1/calibrated_motion_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/calibrated-motion-selection-v1/calibrated_motion_selection/outputs/selection_manifest.json',
        training=ROOT/'reports/experiments/image-motion-linker-v2-training.json',
        native=ROOT/'reports/experiments/independent-joint-selection-v1-score.json',
        causal=ROOT/'reports/experiments/causal-motion-selection-v1-score.json',
        parent=ROOT/'reports/experiments/independent-known-null-selection-v1-score.json',
        flow=ROOT/'reports/experiments/backward-flow-selection-v1-score.json',
        uncalibrated=ROOT/'reports/experiments/image-motion-linker-selection-v1-score.json',
        calibration=ROOT/'reports/experiments/association-calibration-fit-v1.json')
    values={k:json.loads(p.read_text()) for k,p in paths.items()}
    for k in ('native','causal','parent','flow','uncalibrated'):
        values[k]=values[k]['result']
    report=compare(**values)
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/calibrated-motion-selection-v1-score.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','delta_vs_uncalibrated','delta_vs_flow','decision','counts')},indent=2))
