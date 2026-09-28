"""Fixed coefficient ensemble and conservative complete-movie selection gates."""
import numpy as np

ARMS = ('source44', 'source6', 'equal_mean')


def fixed_weights(source44, source6):
    values = [np.asarray(x, np.float64) for x in (source44, source6)]
    if any(x.shape != (30,) or not np.isfinite(x).all() for x in values):
        raise ValueError('Two finite matching 30-feature heads required')
    return dict(source44=values[0].copy(), source6=values[1].copy(),
                equal_mean=(values[0] + values[1]) * .5)


def quality_checks(summaries, per_movie, by_embryo, movies):
    arms = ('submitted8',) + ARMS
    if set(summaries) != set(arms) or set(per_movie) != set(arms):
        raise ValueError('Missing comparison arm')
    if not movies or len(set(movies)) != len(movies):
        raise ValueError('Invalid movie inventory')
    if any(set(per_movie[a]) != set(movies) for a in arms):
        raise ValueError('Incomplete per-movie scoring')
    embryos = {s.split('_')[0] for s in movies}
    if set(by_embryo) != embryos or any(set(v) != set(arms) for v in by_embryo.values()):
        raise ValueError('Incomplete embryo scoring')

    def finite(s, expected):
        return s['n'] == s['n_adj'] == expected and all(
            isinstance(s[k], (float, int)) and np.isfinite(s[k])
            for k in ('score', 'edge_jaccard', 'adj_edge_jaccard', 'node_recall'))

    complete = all(finite(summaries[a], len(movies)) for a in arms)
    complete &= all(finite(s, 1) for v in per_movie.values() for s in v.values())
    complete &= all(finite(v[a], sum(s.startswith(e + '_') for s in movies))
                    for e, v in by_embryo.items() for a in arms)

    def division_ok(base, new):
        if all(sum(s[k] for k in ('division_tp', 'division_fp', 'division_fn')) == 0
               for s in (base, new)):
            return True
        return all(isinstance(s['division_jaccard'], (float, int)) and
                   np.isfinite(s['division_jaccard']) for s in (base, new)) and (
                   new['division_jaccard'] >= base['division_jaccard'] - 1e-12)

    baseline = summaries['submitted8']
    return {a: dict(
        finite_complete_scoring=bool(complete),
        pooled_score_improves=summaries[a]['score'] > baseline['score'] + 1e-12,
        raw_edge_nonregressing=summaries[a]['edge_jaccard'] >= baseline['edge_jaccard'] - 1e-12,
        division_jaccard_nonregressing=division_ok(baseline, summaries[a]),
        every_movie_nonregressing=all(per_movie[a][s]['score'] >= per_movie['submitted8'][s]['score'] - 1e-12 for s in movies),
        every_embryo_nonregressing=all(v[a]['score'] >= v['submitted8']['score'] - 1e-12 and
            v[a]['edge_jaccard'] >= v['submitted8']['edge_jaccard'] - 1e-12 and
            division_ok(v['submitted8'], v[a]) for v in by_embryo.values())) for a in ARMS}


def select_candidate(summaries, checks):
    eligible = [a for a in ARMS if all(checks[a].values())]
    # Stable tie-breaking favors the single models before the coefficient mean.
    return max(eligible, key=lambda a: summaries[a]['score']) if eligible else None
