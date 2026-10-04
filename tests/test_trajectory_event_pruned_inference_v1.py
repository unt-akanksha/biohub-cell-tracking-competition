import numpy as np
import pytest

from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_event_fast_inference_v1 import refine as reference
from research.trajectory_event_pruned_inference_v1 import refine


@pytest.mark.parametrize('seed',range(5))
def test_dominance_full_graph_matches_unreduced_objective_and_graph(seed):
    rng=np.random.default_rng(seed);graph=dict(nodes={},edges=[])
    for t in range(4):
        for j in range(5):
            ident=t*5+j
            graph['nodes'][str(ident)]=dict(node_id=ident,t=t,z=20,y=80+j,x=80+float(rng.random()))
            if t and j<4:graph['edges'].append(dict(source_id=ident-5,target_id=ident))
    groups=candidates(graph,graph);edge=rng.normal(size=(len(groups['parents']),18)).astype(np.float32)
    weights=rng.normal(size=30)
    expected,old=reference(graph,graph,groups,edge,weights)
    actual,new=refine(graph,graph,groups,edge,weights)
    assert expected==actual and not new['solver_fallbacks'] and not new['budget_exhausted']
    for a,b in zip(old['frames'],new['frames']):
        assert a['gain']==pytest.approx(b['gain'],abs=1e-9)
