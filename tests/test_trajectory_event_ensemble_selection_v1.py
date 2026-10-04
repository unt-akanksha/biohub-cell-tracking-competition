from copy import deepcopy
import numpy as np
import pytest
from research.trajectory_event_ensemble_selection_v1 import ARMS, fixed_weights, quality_checks, select_candidate


def fixture():
    row = dict(n=1, n_adj=1, score=.9, edge_jaccard=.9, adj_edge_jaccard=.9,
               node_recall=.98, division_jaccard=.1, division_tp=1, division_fp=4, division_fn=5)
    movies = ['44b6_a', '6bba_b']
    per = {a: {s: deepcopy(row) for s in movies} for a in ('submitted8',) + ARMS}
    totals = {a: dict(row, n=2, n_adj=2) for a in per}
    embryos = {e: {a: deepcopy(row) for a in per} for e in ('44b6', '6bba')}
    for a in ARMS:
        totals[a]['score'] = .91
        for s in movies:
            per[a][s]['score'] = .91
        for e in embryos:
            embryos[e][a]['score'] = .91
    return totals, per, embryos, movies


def test_fixed_mean_has_no_mutation_or_movie_routing():
    a, b = np.arange(30.), np.arange(30.) * 3
    result = fixed_weights(a, b)
    np.testing.assert_array_equal(result['equal_mean'], np.arange(30.) * 2)
    result['source44'][0] = 8
    assert a[0] == 0


@pytest.mark.parametrize('bad', [np.zeros(29), np.full(30, np.nan)])
def test_invalid_member_rejected(bad):
    with pytest.raises(ValueError):
        fixed_weights(np.zeros(30), bad)


def test_complete_gain_eligible_and_stable_selection():
    args = fixture()
    checks = quality_checks(*args)
    assert all(all(v.values()) for v in checks.values())
    assert select_candidate(args[0], checks) == 'source44'


@pytest.mark.parametrize('failure', ['movie', 'embryo', 'raw', 'division', 'missing_count'])
def test_each_failure_blocks_even_with_pooled_gain(failure):
    totals, per, embryos, movies = fixture()
    for a in ARMS:
        if failure == 'movie': per[a][movies[0]]['score'] = .89
        if failure == 'embryo': embryos['44b6'][a]['edge_jaccard'] = .89
        if failure == 'raw': totals[a]['edge_jaccard'] = .89
        if failure == 'division': totals[a]['division_jaccard'] = .09
        if failure == 'missing_count': totals[a]['n_adj'] = 1
    checks = quality_checks(totals, per, embryos, movies)
    assert select_candidate(totals, checks) is None


def test_missing_movie_fails_loudly():
    totals, per, embryos, movies = fixture()
    del per['equal_mean'][movies[0]]
    with pytest.raises(ValueError, match='Incomplete'):
        quality_checks(totals, per, embryos, movies)
