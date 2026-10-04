import numpy as np
import pytest

from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_event_inference_v1 import refine as reference
from research.trajectory_event_fast_inference_v1 import refine


@pytest.mark.parametrize('seed',range(4))
def test_full_movie_edges_exactly_match_reference_for_random_weights(seed):
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
    assert expected==actual and old['frames']==new['frames']
    assert not new['solver_fallbacks'] and not new['budget_exhausted']
