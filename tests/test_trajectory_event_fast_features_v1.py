import numpy as np
import pytest

from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_event_candidates_v1 import frames
from research.trajectory_event_features_v1 import features
from research.trajectory_event_fast_features_v1 import FeatureContext


@pytest.mark.parametrize('seed',range(5))
def test_vectorized_geometry_exactly_matches_reference_with_history_and_future(seed):
    rng=np.random.default_rng(seed)
    graph=dict(nodes={},edges=[])
    for t in range(4):
        for j in range(6):
            ident=t*6+j
            z,y,x=(np.array([20.,80.,80.])+rng.uniform(-3,3,3)).tolist()
            graph['nodes'][str(ident)]=dict(node_id=ident,t=t,z=z,y=y,x=x)
            if t>0 and j<4:graph['edges'].append(dict(source_id=ident-6,target_id=ident))
    groups=candidates(graph,graph)
    edge=rng.normal(size=(len(groups['parents']),18)).astype(np.float32)
    context=FeatureContext(graph,edge)
    for case in frames(graph,graph,groups):
        expected=features(case,graph,edge);actual=context.features(case)
        assert np.array_equal(actual,expected)


def test_invalid_geometry_and_edge_values_are_rejected():
    graph=dict(nodes={'0':dict(t=0,z=1000,y=0,x=0)},edges=[])
    with pytest.raises(ValueError,match='outside'):FeatureContext(graph,np.zeros((1,18)))
    with pytest.raises(ValueError,match='schema'):FeatureContext(graph,np.full((1,18),np.nan))
