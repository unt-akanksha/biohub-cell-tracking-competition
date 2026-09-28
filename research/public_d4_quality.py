"""Predeclared training-diagnostic gates for one fixed D4 geometry correction."""
import math
from research.public_d4_full_movie import STEMS


def compare(rows, summaries, embryos, per_movie_summaries):
    for arm in ('original', 'corrected'):
        if [r['stem'] for r in rows[arm]] != list(STEMS):
            raise ValueError('All four ordered complete movies required')
        if set(embryos[arm]) != {'44b6', '6bba'}:
            raise ValueError('Both embryo diagnostics required')
    delta = {k: summaries['corrected'][k]-summaries['original'][k]
             for k in ('score', 'edge_jaccard', 'adj_edge_jaccard')}
    movie_delta = {s: per_movie_summaries['corrected'][s]['score']-
                  per_movie_summaries['original'][s]['score'] for s in STEMS}
    embryo_delta = {e: embryos['corrected'][e]['score']-embryos['original'][e]['score']
                   for e in ('44b6', '6bba')}
    if not all(math.isfinite(v) for v in [*delta.values(), *movie_delta.values(), *embryo_delta.values()]):
        raise ValueError('Finite complete-movie scores required')
    gates = dict(pooled_combined_strictly_improves=delta['score'] > 0,
                 pooled_raw_edge_jaccard_strictly_improves=delta['edge_jaccard'] > 0,
                 per_embryo_combined_nonregression=min(embryo_delta.values()) >= 0,
                 per_movie_combined_nonregression=min(movie_delta.values()) >= 0)
    return dict(gates=gates, diagnostic_gate_passed=all(gates.values()),
                pooled_delta=delta, per_movie_combined_delta=movie_delta,
                per_embryo_combined_delta=embryo_delta,
                worst_movies={arm: min(per_movie_summaries[arm],
                    key=lambda s: per_movie_summaries[arm][s]['score']) for arm in ('original','corrected')},
                independently_held_out=False, authorized_for_submission=False,
                limitation='Public-checkpoint training overlap; diagnostic pass alone cannot authorize promotion')
