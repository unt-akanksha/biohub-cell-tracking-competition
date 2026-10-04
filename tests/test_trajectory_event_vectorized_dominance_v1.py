import numpy as np
import pytest

from research.trajectory_event_assignment_v1 import problem,solve
from research.trajectory_event_dominance_v1 import allowed_options as reference
from research.trajectory_event_vectorized_dominance_v1 import allowed_options


def make_case(parents=3,children=7):
    options=[(p,-1,-1) for p in range(parents)]+[(-1,c,-1) for c in range(children)]
    options += [(p,c,-1) for p in range(parents) for c in range(children)]
    options += [(p,a,b) for p in range(parents) for a in range(children) for b in range(a+1,children)]
    return problem(options,parents,children)


def test_exact_reference_masks_with_partial_constraints_and_shuffled_options():
    rng=np.random.default_rng(20260914)
    for _ in range(100):
        case=make_case();case=problem(case['options'][rng.permutation(len(case['options']))],3,7)
        scores=rng.normal(size=len(case['options']))
        allowed=rng.random(len(scores))>.2
        old,old_report=reference(case,scores,allowed)
        new,new_report=allowed_options(case,scores,allowed)
        assert np.array_equal(old,new) and old_report==new_report


@pytest.mark.parametrize('delta',[0.,1e-12,1e-10,1e-9,-1e-12])
def test_reference_tie_and_near_tie_arithmetic(delta):
    case=make_case(1,2);scores=np.zeros(len(case['options']))
    scores[-1]=-delta
    old,report=reference(case,scores)
    new,actual=allowed_options(case,scores)
    assert np.array_equal(old,new) and report==actual


def test_forbidden_birth_cannot_prune_known_fork():
    case=make_case(1,2);scores=np.zeros(len(case['options']));scores[-1]=-10
    mask=np.ones(len(scores),bool);mask[case['options'][:,0]<0]=False
    keep,report=allowed_options(case,scores,mask)
    assert keep[-1] and report['dominated_forks']==0


def test_exact_solver_objective_and_choice_on_valid_problem():
    rng=np.random.default_rng(108);case=make_case(2,4)
    for _ in range(5):
        scores=rng.normal(size=len(case['options']))
        old,_=reference(case,scores);new,_=allowed_options(case,scores)
        assert np.array_equal(solve(case,scores,allowed=old),solve(case,scores,allowed=new))


def test_no_available_links_and_invalid_inputs():
    case=make_case();scores=np.zeros(len(case['options']))
    mask=case['options'][:,2]>=0
    old,report=reference(case,scores,mask);new,actual=allowed_options(case,scores,mask)
    assert np.array_equal(old,new) and report==actual
    with pytest.raises(ValueError):allowed_options(case,scores,np.ones(len(scores),int))
    with pytest.raises(ValueError):allowed_options(case,scores+float('nan'))
