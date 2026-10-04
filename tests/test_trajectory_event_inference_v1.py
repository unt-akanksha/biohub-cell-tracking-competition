import copy
import numpy as np
import pytest

from research.trajectory_candidate_inventory_v1 import candidates
import research.trajectory_event_inference_v1 as inference


def graph():
    return dict(nodes={str(i):dict(node_id=i,t=int(i>=2),z=0,y=0,x=i) for i in range(4)},
                edges=[dict(source_id=0,target_id=2)])


def test_full_graph_composition_can_add_fork_without_changing_any_cell():
    g=graph();before=copy.deepcopy(g);groups=candidates(g,g)
    weights=np.zeros(30);weights[19]=5
    result,report=inference.refine(g,g,groups,np.zeros((len(groups['parents']),18)),weights)
    assert g==before and result['nodes']==before['nodes']
    assert len(result['edges'])==2 and report['added_edges']>=1
    assert len({e['source_id'] for e in result['edges']})==1


def test_zero_scores_preserve_exact_original_graph():
    g=graph();groups=candidates(g,g)
    result,report=inference.refine(g,g,groups,np.zeros((len(groups['parents']),18)),np.zeros(30))
    assert result==g and report['added_edges']==report['removed_edges']==0


def test_existing_division_remains_protected():
    g=graph();g['edges'].append(dict(source_id=0,target_id=3));groups=candidates(g,g)
    result,report=inference.refine(g,g,groups,np.zeros((len(groups['parents']),18)),np.zeros(30))
    assert result==g and report['processed_frames']==0


def test_expired_movie_budget_preserves_every_unprocessed_edge(monkeypatch):
    g=graph();groups=candidates(g,g);clock=iter([0.,2.,3.])
    monkeypatch.setattr(inference.time,'monotonic',lambda:next(clock))
    result,report=inference.refine(g,g,groups,np.zeros((len(groups['parents']),18)),np.ones(30),max_seconds=1.)
    assert result==g and report['budget_exhausted'] and report['processed_frames']==0


def test_nonfinite_model_is_not_silently_fallen_back():
    g=graph();groups=candidates(g,g)
    with pytest.raises(ValueError):
        inference.refine(g,g,groups,np.zeros((len(groups['parents']),18)),np.full(30,np.nan))
