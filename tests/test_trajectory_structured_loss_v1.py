import numpy as np
from research.trajectory_structured_loss_v1 import prepare,hinge,fit,infer


def fixture():
    problem=dict(edge_index=np.arange(4).reshape(2,2),parents=np.array([0,1]),
                 group_indices=np.array([0,1]),current_cols=np.array([0,1]))
    matrix=np.array([[0.],[1.],[1.],[0.]])
    return problem,matrix


def test_partial_supervision_drives_joint_cycle_and_does_not_label_unknown_row():
    problem,matrix=fixture()
    case,status=prepare(problem,matrix,np.array([1,-1]),np.array([True,True,False,False]))
    assert status=='prepared' and case['n_constraints']==1
    assert not case['margin'][1].any() and case['constrained'][1].all()
    value,gradient=hinge(case,np.array([0.]))
    assert value==1. and gradient.tolist()==[-2.]
    weights,receipt=fit({'source':[case]},np.zeros(1),steps=100)
    assert receipt['updates']==100 and hinge(case,weights)[0]<value
    assert infer(problem,matrix,weights).tolist()==[1,0]


def test_structured_gradient_matches_finite_difference_away_from_ties():
    problem,matrix=fixture();case,_=prepare(problem,matrix,np.array([1,-1]),np.ones(4,bool))
    w=np.array([.1]);eps=1e-6
    numerical=(hinge(case,w+eps)[0]-hinge(case,w-eps)[0])/(2*eps)
    assert np.allclose(hinge(case,w)[1],numerical,atol=1e-7)


def test_conflicting_division_constraints_are_not_forced_into_one_to_one_loss():
    problem,matrix=fixture()
    case,status=prepare(problem,matrix,np.array([1,1]),np.ones(4,bool))
    assert case is None and status=='no_unique_reachable_constraints'


def test_ambiguous_alternative_has_zero_margin():
    problem,matrix=fixture()
    case,_=prepare(problem,matrix,np.array([1,-1]),np.array([False,True,False,False]))
    assert not case['margin'].any()
