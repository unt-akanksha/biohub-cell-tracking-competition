"""Apply the previously declared source gate, with no score-selected settings."""
import math


def compare(rows, summaries, stems):
    if len(stems) != 8 or len(set(stems)) != 8:
        raise ValueError('Eight unique fixed source movies required')
    for arm in ('parent', 'control', 'candidate'):
        if [r['stem'] for r in rows[arm]] != stems:
            raise ValueError('Exact ordered paired source scope required')
        if any(not math.isfinite(summaries[arm][k]) for k in ('score', 'edge_jaccard', 'node_recall')):
            raise ValueError('Finite complete-source metrics required')
    c, p, s = rows['candidate'], rows['parent'], rows['control']
    if any(a['num_pred_nodes'] != b['num_pred_nodes'] or a['node_recall'] != b['node_recall'] for a, b in zip(c, s)):
        raise ValueError('Static and image-flow arms must preserve identical raw nodes and recall')
    parent = {k: summaries['candidate'][k] - summaries['parent'][k] for k in ('score', 'edge_jaccard', 'node_recall')}
    static = {k: summaries['candidate'][k] - summaries['control'][k] for k in ('score', 'edge_jaccard', 'node_recall')}
    movies = [dict(stem=a['stem'], adjusted_edge_delta=a['adj_edge_jaccard'] - b['adj_edge_jaccard']) for a, b in zip(c, p)]
    if any(not math.isfinite(r['adjusted_edge_delta']) for r in movies):
        raise ValueError('Finite per-movie metrics required')
    worst = min(r['adj_edge_jaccard'] for r in c) - min(r['adj_edge_jaccard'] for r in p)
    conditions = dict(score_gain_over_parent=parent['score'] > 0, raw_edge_gain_over_parent=parent['edge_jaccard'] > 0,
        recall_guard=parent['node_recall'] >= -.005-1e-12,
        five_movies_improve=sum(r['adjusted_edge_delta'] > 0 for r in movies) >= 5,
        per_movie_loss_bounded=min(r['adjusted_edge_delta'] for r in movies) >= -.02-1e-12,
        worst_movie_preserved=worst >= -.01-1e-12,
        score_gain_over_static=static['score'] > 0, raw_edge_gain_over_static=static['edge_jaccard'] > 0)
    return dict(parent_deltas=parent, static_deltas=static, per_movie_deltas=movies, worst_movie_delta=worst,
                conditions=conditions, source_gate_passed=all(conditions.values()), authorized_for_submission=False,
                caveat='Owned flow excludes these movies; FOCUS pretraining overlap remains unverified.')
