from copy import deepcopy
from pathlib import Path
import runpy
import pytest

MODULE=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/score-focus-owned-neural-probe.py'))


def rows():
    values=[dict(stem=s,edge_tp=10,edge_fp=5,edge_fn=2,num_pred_nodes=40,ordinary_tp=8,division_tp=1)
        for s in ('6bba_f1fde7e0','6bba_23af9eeb')]
    return dict(control=values,candidate=deepcopy(values))


def test_tie_rejected_but_same_recall_lower_false_edges_pass():
    data=rows();assert not MODULE['gate'](data)['feasibility_gate_passed']
    data['candidate'][0]['edge_fp']=2
    assert MODULE['gate'](data)['feasibility_gate_passed']


@pytest.mark.parametrize('field',['edge_tp','ordinary_tp','division_tp','num_pred_nodes'])
def test_gain_does_not_override_per_movie_invariants(field):
    data=rows();data['candidate'][0]['edge_fp']=0;data['candidate'][1][field]-=1
    assert not MODULE['gate'](data)['feasibility_gate_passed']


def test_wrong_scope_rejected():
    data=rows();data['candidate'].reverse()
    with pytest.raises(ValueError): MODULE['gate'](data)
