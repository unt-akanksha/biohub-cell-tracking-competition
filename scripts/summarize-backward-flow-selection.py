"""Compare the image-motion result with all strongest fixed-node controls."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(result,manifest_bytes,fit,native,causal,static,learned):
    manifest = json.loads(manifest_bytes)
    if (result['flow_checkpoint_sha256'] != fit['checkpoint_sha256']
        or manifest['checkpoint_sha256'] != fit['checkpoint_sha256']
        or fit['small_fit_gate_passed'] is not True or fit['steps'] != 1000
        or result['flow_manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest()
        or result['checkpoint_sha256'] != native['checkpoint_sha256']
        or result['target_audit_opened'] is not False or result['authorized_for_submission'] is not False
        or result['variance_um2'] != static['variance_um2'] or result['null_logit'] != static['null_logit']):
        raise ValueError('Exact flow fit, native detector and static policy required')
    rows = result['per_movie']['motion']
    if result['per_movie']['original'] != native['per_movie']:
        raise ValueError('Native scoring did not reproduce')
    controls = dict(native=native['per_movie'],causal=causal['per_movie']['motion'],
                    static=static['per_movie']['motion'],learned=learned['per_movie'])
    for name,old in controls.items():
        if [r['stem'] for r in rows] != [r['stem'] for r in old]:
            raise ValueError('Comparison movie coverage changed: '+name)
        for row,previous in zip(rows,old):
            if any(row[k] != previous[k] for k in ('num_pred_nodes','node_recall','total_node_ratio')):
                raise ValueError('Comparison detections changed: '+name)
            receipt = result['receipts'][row['stem']]
            if receipt['nodes_unchanged'] is not True or receipt['predicted_nodes'] != row['num_pred_nodes']:
                raise ValueError('Missing exact node preservation')
    scores = dict(native=native['summary']['score'],causal=causal['summaries']['motion']['score'],
                  static=static['summaries']['motion']['score'],learned=learned['summary']['score'])
    score = result['summaries']['motion']['score']
    if not math.isclose(result['score_delta'],score-scores['native'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('Native score delta mismatch')
    keys = ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')
    counts = {name:{k:sum(r[k] for r in values) for k in keys} for name,values in dict(candidate=rows,**controls).items()}
    return dict(status='verified_backward_flow_selection',result=result,
        deltas={name:score-value for name,value in scores.items()},counts=counts,
        adjusted_edge_regressions={name:[r['stem'] for r,p in zip(rows,previous)
            if r['adj_edge_jaccard'] < p['adj_edge_jaccard']-1e-12] for name,previous in controls.items()},
        worst_movie=min(rows,key=lambda r:r['adj_edge_jaccard']),authorized_for_submission=False,
        decision='source_selection_gain_requires_embryo_audit' if score>max(scores.values()) else 'not_strongest_standalone',
        caveat='Flow vs static isolates predicted displacement with identical nodes, variance, null and topology; source selection only')


if __name__ == '__main__':
    paths = dict(result=ROOT/'.biohub/cache/kernel-outputs/backward-flow-scoring-v1/backward_flow_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/backward-flow-selection-v1/backward_flow_selection/outputs/flow_manifest.json',
        fit=ROOT/'reports/experiments/backward-flow-fit-v1-result.json',
        native=ROOT/'reports/experiments/independent-joint-selection-v1-score.json',
        causal=ROOT/'reports/experiments/causal-motion-selection-v1-score.json',
        static=ROOT/'.biohub/cache/kernel-outputs/joint-static-motion-selection-v1/joint_static_motion_score/static_motion_score.json',
        learned=ROOT/'reports/experiments/independent-known-null-selection-v1-score.json')
    values = {k:json.loads(p.read_text()) for k,p in paths.items() if k!='manifest'}
    for name in ('native','causal','learned'):
        values[name] = values[name]['result']
    report = compare(**values,manifest_bytes=paths['manifest'].read_bytes())
    report['source_sha256'] = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    (ROOT/'reports/experiments/backward-flow-selection-v1-score.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','deltas','counts','decision')},indent=2))
