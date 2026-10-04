from copy import deepcopy
from pathlib import Path
import runpy

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/audit-trajectory-event-anchor-selection-v1.py'))


def fixture():
    base=dict(n=1,n_adj=1,score=.9,edge_jaccard=.85,adj_edge_jaccard=.85,
              node_recall=.99,division_tp=1,division_fp=1,division_fn=1)
    new=dict(base,score=.91,edge_jaccard=.86,adj_edge_jaccard=.86)
    totals=dict(baseline=base,anchor=new)
    return totals,{a:{'movie':deepcopy(v)} for a,v in totals.items()},{'embryo':deepcopy(totals)}


def test_valid_gain():
    assert all(MODULE['quality_checks'](*fixture()).values())


def test_missing_scored_movie_fails():
    args=fixture();args[1]['anchor']['movie']['n_adj']=0
    assert not MODULE['quality_checks'](*args)['finite_complete_scoring']


def test_movie_regression_is_not_hidden_by_pooled_gain():
    args=fixture();args[1]['anchor']['movie']['score']=.89
    assert not MODULE['quality_checks'](*args)['every_movie_nonregressing']


def test_embryo_and_division_failure():
    args=fixture();args[2]['embryo']['anchor']['score']=.89
    args[0]['anchor']['division_fp']=2
    checks=MODULE['quality_checks'](*args)
    assert not checks['every_embryo_nonregressing'] and not checks['division_counts_nonregressing']


def test_nonfinite_fails():
    args=fixture();args[1]['anchor']['movie']['score']=float('nan')
    assert not MODULE['quality_checks'](*args)['finite_complete_scoring']
