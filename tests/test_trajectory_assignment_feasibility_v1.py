import numpy as np
from scipy.optimize import linear_sum_assignment
from research.trajectory_assignment_feasibility_v1 import problems,counts_only
from research.trajectory_joint_assignment_v1 import apply


def fixture():
    graph=dict(nodes={i:dict(t=int(i>=2),z=0,y=0,x=i) for i in range(4)},
               edges=[dict(source_id=0,target_id=2),dict(source_id=1,target_id=3)])
    groups=dict(children=np.array([2,3]),parents=np.array([0,1,0,1]),
                offsets=np.array([0,2,4]),current=np.array([0,1]))
    return graph,groups


def test_joint_upper_bound_can_require_a_cycle_not_independent_changes():
    graph,groups=fixture()
    result=counts_only(graph,groups,np.array([1,0]),np.ones(4,bool))
    assert result['current_correct']==0 and result['constrained_upper_bound_correct']==2
    assert not result['oracle_predictions_exported']


def test_protected_division_has_no_false_recoverability():
    graph,groups=fixture();graph['nodes'][4]=dict(t=1,z=0,y=0,x=4)
    graph['edges'].append(dict(source_id=0,target_id=4))
    result=counts_only(graph,groups,np.array([1,0]),np.ones(4,bool))
    assert result['constrained_upper_bound_correct']==0


def test_feasible_problems_reproduce_frozen_policy_assignment():
    graph,groups=fixture();scores=np.array([0.,2.,2.,0.])
    problem=list(problems(graph,groups))[0]
    row,col=linear_sum_assignment(-scores[problem['edge_index']])
    result,_=apply(graph,groups,scores[:,None],np.ones(1))
    assert [(e['source_id'],e['target_id']) for e in result['edges']]==[
        (int(problem['parents'][b]),int(problem['children'][a])) for a,b in zip(row,col)]
