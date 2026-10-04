import numpy as np
import pytest

from research.trajectory_event_assignment_v1 import problem,prepare as reference_prepare,hinge as reference_hinge
from research.trajectory_event_fast_training_v1 import prepare,hinge


def tiny():
    return problem([(0,-1,-1),(0,0,-1),(0,1,-1),(0,0,1),(-1,0,-1),(-1,1,-1)],1,2)


@pytest.mark.parametrize('targets',[[0,0],[0,-1],[-1,-1]])
def test_constructive_partial_labels_match_reference(targets):
    p=tiny();x=np.eye(6);targets=np.array(targets);safe=np.ones((6,2),bool)
    a,ar=reference_prepare(p,x,targets,safe);b,br=prepare(p,x,targets,safe)
    assert ar==br
    if a is not None:
        for key in ('x','allowed','margin'):assert np.array_equal(a[key],b[key])
        assert a['n_constraints']==b['n_constraints']


def test_constructive_rejects_unreachable_fork_not_unknown_birth():
    p=problem([(0,-1,-1),(0,0,-1),(0,1,-1),(-1,0,-1),(-1,1,-1)],1,2)
    assert prepare(p,np.eye(5),np.array([0,0]),np.zeros((5,2),bool))[1]=='incompatible_partial_constraints'


def test_reduced_hinge_and_gradient_match_away_from_ties():
    p=tiny();case,_=prepare(p,np.eye(6),np.array([0,0]),np.zeros((6,2),bool))
    rng=np.random.default_rng(20260914)
    for _ in range(20):
        weights=rng.normal(size=6)
        a,ag=reference_hinge(case,weights);b,bg=hinge(case,weights)
        assert abs(a-b)<1e-9 and np.array_equal(ag,bg)
