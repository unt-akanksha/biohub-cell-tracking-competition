import numpy as np
from research.trajectory_joint_assignment_v1 import apply


def graph():
    return dict(nodes={i:dict(t=int(i>=2),z=0,y=0,x=i) for i in range(4)},
                edges=[dict(source_id=0,target_id=2),dict(source_id=1,target_id=3)])


def groups():
    return dict(children=np.array([2,3]),parents=np.array([0,1,0,1]),
                offsets=np.array([0,2,4]),current=np.array([0,1]))


def test_joint_swap_preserves_degrees_and_does_not_mutate_input():
    original=graph()
    result,details=apply(original,groups(),np.array([[0.],[2.],[2.],[0.]]),np.array([1.]))
    assert result['edges']==[dict(source_id=1,target_id=2),dict(source_id=0,target_id=3)]
    assert original==graph() and details['changed_edges']==2 and details['degrees_unchanged']


def test_independent_greedy_collision_resolved_without_division():
    result,details=apply(graph(),groups(),np.array([[0.],[10.],[0.],[20.]]),np.array([1.]))
    assert result==graph() and details['changed_edges']==0


def test_existing_division_incident_edges_protected():
    original=graph();original['nodes'][4]=dict(t=1,z=0,y=0,x=4)
    original['edges'].append(dict(source_id=0,target_id=4))
    result,details=apply(original,groups(),np.array([[0.],[10.],[10.],[0.]]),np.array([1.]))
    assert result==original and details['changed_edges']==0


def test_equal_cost_is_exact_noop():
    original=graph()
    result,details=apply(original,groups(),np.zeros((4,1)),np.ones(1))
    assert result==original and details['changed_edges']==0


def test_gap_incident_nodes_are_protected():
    original=graph();original['nodes'][4]=dict(t=3,z=0,y=0,x=4)
    original['edges'].append(dict(source_id=2,target_id=4))
    result,details=apply(original,groups(),np.array([[0.],[10.],[10.],[0.]]),np.array([1.]))
    assert result==original and details['changed_edges']==0
