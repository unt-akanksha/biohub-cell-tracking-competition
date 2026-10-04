"""Exposed-target diagnostic comparison; cannot pass the original source gate."""
import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
G = runpy.run_path(str(ROOT/'research/owned_detector_transfer_contract.py'))
BASE_SHA = 'd59568f8f69b700a29af62f128cac27ca1b524b2a57f8cad1c7e403d6d6890fa'


def compare(score, manifest, baseline, split):
    expected = G['contract'](split)
    if (score['status']!='scored_exposed_transfer_diagnostic' or manifest['status']!='completed'
        or baseline['status']!='verified_initial_embryo_audit'
        or any(x.get('owned_transfer_diagnostic')!=expected for x in (score,manifest))
        or any(x['checkpoint_sha256']!=G['CHECKPOINT_SHA'] for x in (score,manifest))
        or any(x['authorized_for_submission'] is not False or x['target_audit_opened'] is not True for x in (score,manifest))
        or manifest['ground_truth_opened'] is not False
        or any(r['processed_frames']!=100 or r['image_shape'][0]!=100 for r in manifest['records'])):
        raise ValueError('Complete frozen exposed-target diagnostic required')
    reference = baseline['result']
    rows = score['per_movie']; control = reference['per_movie']
    if any([r['stem'] for r in values]!=G['STEMS'] for values in (rows,control,manifest['records'])):
        raise ValueError('Exact same four complete movies required')
    fields = ('score','edge_jaccard','node_recall')
    if any(not math.isfinite(x['summary'][k]) for x in (score,reference) for k in fields):
        raise ValueError('Finite complete metrics required')
    deltas = {k:score['summary'][k]-reference['summary'][k] for k in fields}
    movies = [dict(stem=r['stem'],adjusted_edge_delta=r['adj_edge_jaccard']-b['adj_edge_jaccard'],
                   recall_delta=r['node_recall']-b['node_recall']) for r,b in zip(rows,control)]
    worst = min(r['adj_edge_jaccard'] for r in rows)-min(r['adj_edge_jaccard'] for r in control)
    conditions = dict(score_gain=deltas['score']>0,raw_edge_gain=deltas['edge_jaccard']>0,
        mean_recall_preserved=deltas['node_recall']>=-.005-1e-12,
        three_movies_improve=sum(r['adjusted_edge_delta']>0 for r in movies)>=3,
        per_movie_loss_bounded=min(r['adjusted_edge_delta'] for r in movies)>=-.02-1e-12,
        worst_movie_preserved=worst>=-.01-1e-12)
    counts = ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes')
    return dict(status='verified_exposed_target_transfer_diagnostic',result=score,
        summary_deltas=deltas,per_movie_deltas=movies,worst_movie_delta=worst,
        diagnostic_conditions=conditions,diagnostic_transfer_supported=all(conditions.values()),
        counts={name:{k:sum(r[k] for r in values) for k in counts} for name,values in [('baseline',control),('pu',rows)]},
        source_combined_gate_passed=False,authorized_for_submission=False,new_target_movies_opened=0,
        caveat='These four target movies were previously exposed. This diagnostic cannot establish independent generalization, reverse the failed source paired gate, or authorize submission.')


if __name__=='__main__':
    paths = dict(score=ROOT/'.biohub/cache/kernel-outputs/owned-detector-pu-transfer-v1-scoring/owned_detector_pu_transfer_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/owned-detector-pu-transfer-v1/owned_detector_pu_transfer/outputs/selection_manifest.json',
        baseline=ROOT/'reports/experiments/detector-spatial-tta-audit-v1-score.json',
        split=ROOT/'research/independent_real_baseline_v1_split.json',
        source=ROOT/'reports/experiments/owned-detector-selection-v1-comparison.json')
    if hashlib.sha256(paths['baseline'].read_bytes()).hexdigest()!=BASE_SHA:
        raise ValueError('Original frozen target baseline changed')
    G['verify'](paths['source'].read_bytes(),paths['split'].read_bytes())
    report=compare(*(json.loads(paths[k].read_text()) for k in ('score','manifest','baseline','split')))
    report['source_sha256']={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/owned-detector-pu-transfer-v1-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='result'},indent=2))
