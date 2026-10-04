import numpy as np

from research.trajectory_event_assignment_v1 import problem,prepare,solve
from research.trajectory_event_dominance_v1 import allowed_options


def tiny():
    return problem([(0,-1,-1),(0,0,-1),(0,1,-1),(0,0,1),(-1,0,-1),(-1,1,-1)],1,2)


def test_dominated_fork_replaced_without_changing_consumed_cells():
    case=tiny();scores=np.array([0.,3.,2.,1.,0.,0.])
    keep,report=allowed_options(case,scores)
    assert not keep[3] and report['dominated_forks']==1
    assert np.array_equal(solve(case,scores),solve(case,scores,allowed=keep))


def test_known_division_not_pruned_using_forbidden_birth():
    case=tiny();prepared,_=prepare(case,np.eye(6),np.array([0,0]),np.zeros((6,2),bool))
    keep,report=allowed_options(case,np.array([0.,3.,2.,-6.,0.,0.]),prepared['allowed'])
    assert keep[3] and report['dominated_forks']==0


def test_ties_are_preserved():
    keep,report=allowed_options(tiny(),np.array([0.,3.,2.,3.,0.,0.]))
    assert keep[3] and report['dominated_forks']==0


def test_optimum_preserved_for_competing_forks_and_partial_masks():
    options=[(p,-1,-1) for p in range(2)]+[(-1,c,-1) for c in range(3)]
    options += [(p,c,-1) for p in range(2) for c in range(3)]
    options += [(p,a,b) for p in range(2) for a in range(3) for b in range(a+1,3)]
    case=problem(options,2,3);rng=np.random.default_rng(20260914)
    for iteration in range(25):
        scores=rng.normal(size=len(options));allowed=np.ones(len(options),bool)
        if iteration%2:allowed[2]=False
        keep,_=allowed_options(case,scores,allowed)
        original=solve(case,scores,allowed=allowed);reduced=solve(case,scores,allowed=keep)
        assert np.isclose(scores@original,scores@reduced,atol=1e-8,rtol=0)
