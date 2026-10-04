"""Archive full-movie score/deltas; never authorize submission from this fold."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(result, control):
    expected = json.loads((ROOT / 'research/independent_real_baseline_v1_split.json').read_text())['folds'][0]['selection']
    if (result['status'] != 'scored_complete_selection'
        or result['checkpoint_sha256'] != 'dbedcb46b1f3684ae31bbe61db55adc7f48c8a2e4a62ba1cf1d7c4d3d844db95'
        or result['target_audit_opened'] is not False or result['authorized_for_submission'] is not False):
        raise ValueError('Unexpected scored checkpoint or scope')
    for scored in (result, control):
        if len(scored['per_movie']) != 8 or {r['stem'] for r in scored['per_movie']} != set(expected):
            raise ValueError('Exact complete source-selection coverage required')
    old = {r['stem']:r for r in control['per_movie']}
    fields = ('edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','node_recall','adj_edge_jaccard','num_pred_nodes')
    deltas = [dict(stem=r['stem'], **{key:r[key]-old[r['stem']][key] for key in fields}) for r in result['per_movie']]
    return dict(result=result, control_summary=control['summary'], per_movie_deltas=deltas,
        score_delta=result['summary']['score']-control['summary']['score'],
        adjusted_edge_regressions=[r['stem'] for r in deltas if r['adj_edge_jaccard'] < -1e-12],
        worst_movie=min(result['per_movie'], key=lambda r:r['adj_edge_jaccard']),
        totals={key:sum(r[key] for r in result['per_movie']) for key in fields[:6]},
        authorization='No production authorization: complete-movie source selection is not embryo-held-out audit',
        authorized_for_submission=False)


if __name__ == '__main__':
    path = ROOT / '.biohub/cache/kernel-outputs/independent-extended-scoring-v1/independent_extended_score/selection_score.json'
    result = json.loads(path.read_text())
    control = json.loads((ROOT / 'reports/experiments/independent-joint-selection-v1-score.json').read_text())['result']
    report = summarize(result,control)
    report['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / 'reports/experiments/independent-extended-selection-v1-score.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('score_delta','adjusted_edge_regressions','totals','worst_movie')},indent=2))
