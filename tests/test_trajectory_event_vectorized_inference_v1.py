import numpy as np

from research.trajectory_candidate_inventory_v1 import candidates
from research import trajectory_event_dominance_v1 as original
from research.trajectory_event_fork16_inference_v1 import refine as reference
from research.trajectory_event_vectorized_inference_v1 import refine,infer


def test_adapter_does_not_mutate_reference_namespaces():
    assert reference.__globals__['infer'] is original.infer
    assert infer.__globals__['allowed_options'] is not original.allowed_options
    assert refine.__code__ is reference.__code__
    assert infer.__code__ is original.infer.__code__


def test_complete_graphs_and_solver_records_exact_for_zero_and_fork_scores():
    graph=dict(nodes={str(i):dict(node_id=i,t=int(i>0),z=0,y=0,x=i) for i in range(11)},
               edges=[dict(source_id=0,target_id=1)])
    groups=candidates(graph,graph);edge=np.zeros((len(groups['parents']),18))
    for gain in (0.,100.):
        weights=np.zeros(30);weights[19]=gain
        old,old_details=reference(graph,graph,groups,edge,weights)
        new,new_details=refine(graph,graph,groups,edge,weights)
        assert old==new
        assert {k:v for k,v in old_details.items() if k!='seconds'}=={k:v for k,v in new_details.items() if k!='seconds'}
