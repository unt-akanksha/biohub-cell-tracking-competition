import numpy as np
from research.trajectory_candidate_inventory_v1 import candidates
from research.trajectory_event_fork16_inference_v1 import refine, FEATURES
from research.trajectory_event_release_v1 import check_event_stage


def graph():
    return dict(nodes={str(i):dict(node_id=i,t=int(i>0),z=0,y=0,x=i) for i in range(11)},
                edges=[dict(source_id=0,target_id=1)])


def test_uniform_expanded_inference_and_zero_gain_identity():
    initial=graph();groups=candidates(initial,initial)
    result,details=refine(initial,initial,groups,np.zeros((len(groups['parents']),18)),np.zeros(len(FEATURES)))
    assert result==initial and details['processed_frames']==1
    assert details['frames'][0]['options']==66
    assert not details['solver_fallbacks'] and not details['budget_exhausted']
    check_event_stage(initial,initial,result,details,frames=2)


def test_positive_fork_score_keeps_graph_invariants():
    initial=graph();groups=candidates(initial,initial)
    weights=np.zeros(len(FEATURES));weights[19]=100
    result,details=refine(initial,initial,groups,np.zeros((len(groups['parents']),18)),weights)
    assert result['nodes']==initial['nodes'] and len(result['edges'])==2
    assert not details['solver_fallbacks'] and not details['budget_exhausted']
    check_event_stage(initial,initial,result,details,frames=2)
