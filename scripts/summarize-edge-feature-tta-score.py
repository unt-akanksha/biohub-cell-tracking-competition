"""Verify and report a feature-only comparison on the exact same detections."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'c5023345d31d91929a8d05219310a9edf1aeecf576d7593cbc5e65208c76b470'


def summarize(candidate, control, manifest, movies):
    for result in (candidate, control):
        if (result['status'] != 'scored_complete_selection' or result['checkpoint_sha256'] != SOURCE_SHA
            or result['target_audit_opened'] is not False or result['authorized_for_submission'] is not False
            or [r['stem'] for r in result['per_movie']] != movies):
            raise ValueError('Exact same-checkpoint complete selection required')
    if (manifest['checkpoint_sha256'] != SOURCE_SHA or manifest['status'] != 'completed'
        or [r['stem'] for r in manifest['records']] != movies
        or any(r.get('reference_nodes_identical') is not True for r in manifest['records'])
        or manifest['edge_feature_tta']['views'] != 8
        or manifest['edge_feature_tta']['maximum_mean_absolute_feature_delta'] <= 0):
        raise ValueError('Missing feature/node-preservation evidence')
    deltas = []
    fields = ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','edge_jaccard','adj_edge_jaccard')
    for current, previous in zip(candidate['per_movie'],control['per_movie']):
        for key in ('num_pred_nodes','node_recall','total_node_ratio'):
            if not math.isclose(current[key],previous[key],rel_tol=0,abs_tol=1e-12):
                raise ValueError('Feature-only comparison changed detection evidence: '+key)
        deltas.append(dict(stem=current['stem'],**{key:current[key]-previous[key] for key in fields}))
    return dict(status='verified_feature_only_comparison',result=candidate,control_summary=control['summary'],
        score_delta=candidate['summary']['score']-control['summary']['score'],per_movie_deltas=deltas,
        detection_metrics_identical=True,feature_receipt=manifest['edge_feature_tta'],
        node_reference_manifest_sha256=manifest['node_reference_manifest_sha256'],
        adjusted_edge_regressions=[r['stem'] for r in deltas if r['adj_edge_jaccard'] < -1e-12],
        scope='Source-embryo selection only; no target-embryo audit or production authorization',
        authorized_for_submission=False)


if __name__ == '__main__':
    score_path = ROOT/'.biohub/cache/kernel-outputs/edge-feature-tta-scoring-v1/edge_feature_tta_score/selection_score.json'
    manifest_path = ROOT/'.biohub/cache/kernel-outputs/edge-feature-tta-selection-v1/edge_feature_tta_selection/outputs/selection_manifest.json'
    control_path = ROOT/'reports/experiments/independent-joint-selection-v1-score.json'
    movies = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]['selection']
    result = summarize(json.loads(score_path.read_text()),json.loads(control_path.read_text())['result'],
                       json.loads(manifest_path.read_text()),movies)
    result['source_sha256'] = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                              [('score',score_path),('manifest',manifest_path),('control',control_path)]}
    (ROOT/'reports/experiments/edge-feature-tta-selection-v1-score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','score_delta','adjusted_edge_regressions','per_movie_deltas')},indent=2))
