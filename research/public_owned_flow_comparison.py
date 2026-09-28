"""Predeclared comparison for a cached public-base continuation repair."""
import math


def compare(rows, summaries):
    candidate = rows['candidate']
    result = {}
    for arm in ('public_control', 'prior_consensus'):
        base = rows[arm]
        if not candidate or [r['stem'] for r in candidate] != [r['stem'] for r in base]:
            raise ValueError('Same ordered complete movie scope required')
        for c, b in zip(candidate, base):
            if c['num_pred_nodes'] != b['num_pred_nodes'] or c['node_recall'] != b['node_recall']:
                raise ValueError('Exact node preservation and recall required')
        delta = {k: summaries['candidate'][k] - summaries[arm][k]
                 for k in ('score', 'edge_jaccard')}
        movies = {c['stem']: c['adj_edge_jaccard'] - b['adj_edge_jaccard']
                  for c, b in zip(candidate, base)}
        if not all(math.isfinite(v) for v in [*delta.values(), *movies.values()]):
            raise ValueError('Finite scores required')
        counts = {k: sum(c[k] - b[k] for c, b in zip(candidate, base))
                  for k in ('edge_tp', 'edge_fp', 'edge_fn')}
        passed = (delta['score'] > 0 and delta['edge_jaccard'] > 0
                  and counts['edge_tp'] > 0 and counts['edge_fp'] <= 0
                  and min(movies.values()) >= 0)
        result[arm] = dict(delta=delta, count_delta=counts,
                           per_movie_adjusted_delta=movies, passed=passed)
    return dict(comparisons=result, diagnostic_gate_passed=all(r['passed'] for r in result.values()),
                authorized_for_submission=False)
