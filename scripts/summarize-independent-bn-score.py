"""Archive exact full-movie normalization comparison; no metric selection."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    path = ROOT / '.biohub/cache/kernel-outputs/independent-bn-scoring-v1/independent_bn_score/selection_score.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    control = json.loads((ROOT / 'reports/experiments/independent-joint-selection-v1-score.json').read_text())['result']
    assert result['status'] == 'scored_complete_selection'
    assert result['checkpoint_sha256'] == '143fd7cd861a854a61dca590405146dd9d4da76b38d2a66bf473f16b84e5446b'
    assert not result['target_audit_opened'] and not result['authorized_for_submission']
    old = {row['stem']: row for row in control['per_movie']}
    assert len(old) == len(result['per_movie']) == 8
    assert set(old) == {row['stem'] for row in result['per_movie']}
    deltas = [dict(stem=row['stem'], **{key: row[key] - old[row['stem']][key] for key in
              ('adj_edge_jaccard', 'node_recall', 'edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp')})
              for row in result['per_movie']]
    return dict(result=result, control_summary=control['summary'], per_movie_deltas=deltas,
                score_delta=result['summary']['score'] - control['summary']['score'],
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decision='Rejected: pooled score falls and six of eight adjusted movie scores regress',
                authorized_for_submission=False)


if __name__ == '__main__':
    report = summarize()
    destination = ROOT / 'reports/experiments/independent-bn-selection-v1-score.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=report['result']['summary'], delta=report['score_delta'], decision=report['decision'])))
