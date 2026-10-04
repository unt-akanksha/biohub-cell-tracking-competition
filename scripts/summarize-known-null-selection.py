"""Same-detection comparison of the completed known-null fit and controls."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(candidate, manifest, training, native, causal):
    if (candidate['status'] != 'scored_complete_selection' or manifest['status'] != 'completed'
            or candidate['checkpoint_sha256'] != training['checkpoint_sha256']
            or manifest['checkpoint_sha256'] != training['checkpoint_sha256']
            or training['steps'] != 1000 or training['detector_unchanged'] is not True
            or training['probe_inputs_replayed'] is not True
            or manifest.get('known_null_training') != dict(version=1,absence_radius_um=7.,unknown_columns_supervised=False)
            or not manifest.get('node_reference_manifest_sha256')
            or any(r.get('reference_nodes_identical') is not True for r in manifest['records'])
            or any(candidate[k] is not False for k in ('target_audit_opened','authorized_for_submission'))):
        raise ValueError('Complete checkpoint-bound frozen-detector selection required')
    stems = [r['stem'] for r in native['per_movie']]
    if len(stems) != 8 or any([r['stem'] for r in rows] != stems for rows in (
            candidate['per_movie'],manifest['records'],causal['per_movie']['motion'])):
        raise ValueError('Exact complete source-selection coverage required')
    if native['checkpoint_sha256'] != 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470':
        raise ValueError('Wrong native reference checkpoint')
    for row, baseline, motion in zip(candidate['per_movie'],native['per_movie'],causal['per_movie']['motion']):
        for key in ('num_pred_nodes','node_recall','total_node_ratio'):
            if any(not math.isclose(row[key], other[key],rel_tol=0,abs_tol=1e-12) for other in (baseline,motion)):
                raise ValueError('Detection metrics changed: '+key)
    fields = ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn')
    totals = {name:{key:sum(row[key] for row in rows) for key in fields}
              for name,rows in [('candidate',candidate['per_movie']),('native',native['per_movie']),
                                ('causal',causal['per_movie']['motion'])]}
    score = candidate['summary']['score']
    return dict(status='verified_known_null_selection',result=candidate,counts=totals,
        delta_vs_native=score-native['summary']['score'],
        delta_vs_causal=score-causal['summaries']['motion']['score'],
        native_summary=native['summary'],causal_summary=causal['summaries']['motion'],
        adjusted_edge_regressions=[r['stem'] for r,b in zip(candidate['per_movie'],native['per_movie'])
                                  if r['adj_edge_jaccard'] < b['adj_edge_jaccard']-1e-12],
        worst_movie=min(candidate['per_movie'],key=lambda r:r['adj_edge_jaccard']),
        detection_metrics_identical=True,authorized_for_submission=False,
        caveat='Extra frozen-linker optimization plus known-null supervision; not an isolated loss ablation')


if __name__ == '__main__':
    paths = dict(candidate=ROOT/'.biohub/cache/kernel-outputs/independent-known-null-scoring-v1/independent_known_null_score/selection_score.json',
        manifest=ROOT/'.biohub/cache/kernel-outputs/independent-known-null-selection-v1/independent_known_null_selection/outputs/selection_manifest.json',
        training=ROOT/'reports/experiments/independent-known-null-v2-training.json',
        native=ROOT/'reports/experiments/independent-joint-selection-v1-score.json',
        causal=ROOT/'reports/experiments/causal-motion-selection-v1-score.json')
    values = {key:json.loads(path.read_text()) for key,path in paths.items()}
    values['native'] = values['native']['result']
    values['causal'] = values['causal']['result']
    result = compare(**values)
    result['source_sha256'] = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    (ROOT/'reports/experiments/independent-known-null-selection-v1-score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','delta_vs_native','delta_vs_causal','counts','adjusted_edge_regressions')},indent=2))
