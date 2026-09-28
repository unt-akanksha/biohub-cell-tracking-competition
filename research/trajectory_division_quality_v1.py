"""Unchanged clean micro/macro nonregression requirements for eight movies."""
import math


def check(all_rows, summaries, embryos, movies, division_stems):
    base, repaired = {r['stem']:r for r in all_rows['original']}, {r['stem']:r for r in all_rows['repaired']}
    if len(base) != 8 or set(base) != set(repaired) or not set(division_stems) <= set(base):
        raise ValueError('All eight complete paired movies required')
    for arm in all_rows:
        if len(all_rows[arm]) != 8 or len({r['stem'] for r in all_rows[arm]}) != 8:
            raise ValueError('Duplicated or missing movie')
    movie_delta = {s: movies['repaired'][s]['score'] - movies['original'][s]['score'] for s in base}
    embryo_delta = {e: embryos['repaired'][e]['score'] - embryos['original'][e]['score'] for e in ('44b6','6bba')}
    score_delta = summaries['repaired']['score'] - summaries['original']['score']
    edge_delta = summaries['repaired']['edge_jaccard'] - summaries['original']['edge_jaccard']
    if not all(math.isfinite(v) for v in [*movie_delta.values(), *embryo_delta.values(), score_delta, edge_delta]):
        raise ValueError('Finite metrics required')
    divisions_covered = {e: sum(base[s]['division_tp'] + base[s]['division_fn']
                              for s in division_stems if s.startswith(e + '_')) for e in ('44b6','6bba')}
    division_safe = all(repaired[s]['division_tp'] >= base[s]['division_tp']
                        and repaired[s]['division_fp'] <= base[s]['division_fp']
                        and repaired[s]['division_fn'] <= base[s]['division_fn'] for s in base)
    tp_gain = sum(r['edge_tp'] for r in repaired.values()) - sum(r['edge_tp'] for r in base.values())
    gates = dict(pooled_combined_strictly_improves=score_delta > 0,
                 pooled_raw_edge_strictly_improves=edge_delta > 0,
                 all_movies_nonregress=min(movie_delta.values()) >= 0,
                 both_embryos_nonregress=min(embryo_delta.values()) >= 0,
                 division_counts_nonregress=division_safe,
                 division_positive_both_embryos=min(divisions_covered.values()) > 0,
                 genuine_edge_tp_increase=tp_gain > 0)
    return dict(gates=gates, diagnostic_gate_passed=all(gates.values()),
                pooled_score_delta=score_delta, pooled_raw_edge_delta=edge_delta,
                edge_tp_gain=tp_gain, per_movie_delta=movie_delta, per_embryo_delta=embryo_delta,
                annotated_divisions_by_embryo=divisions_covered,
                worst_movies={a:min(movies[a], key=lambda s:movies[a][s]['score']) for a in movies},
                independently_held_out=False, authorized_for_submission=False)
