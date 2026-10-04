import numpy as np

from research.trajectory_event_assignment_v1 import problem
from research.trajectory_event_fast_training_v1 import prepare,hinge as reference
from research.trajectory_event_adam_state_v1 import initialize,update
from research.trajectory_event_vectorized_training_v1 import hinge
from research.trajectory_event_dominance_v1 import allowed_options as original_pruning


def test_private_hinge_keeps_original_training_namespace():
    assert reference.__globals__['allowed_options'] is original_pruning
    assert hinge.__code__ is reference.__code__
    assert hinge.__globals__ is not reference.__globals__


def test_partial_division_hinge_and_optimizer_states_reproduce_exactly():
    options=[(p,-1,-1) for p in range(2)]+[(-1,c,-1) for c in range(4)]
    options += [(p,c,-1) for p in range(2) for c in range(4)]
    options += [(p,a,b) for p in range(2) for a in range(4) for b in range(a+1,4)]
    raw=problem(options,2,4);rng=np.random.default_rng(147)
    case,reason=prepare(raw,rng.normal(size=(len(options),30)),np.array([0,0,1,-1]),np.ones((len(options),2),bool))
    assert reason=='prepared'
    anchor=rng.normal(size=30);old=initialize(anchor);new=initialize(anchor)
    for _ in range(6):
        a,ga=reference(case,old['weights']);b,gb=hinge(case,new['weights'])
        assert a==b and np.array_equal(ga,gb)
        update(old,ga+.1*(old['weights']-anchor),.03)
        update(new,gb+.1*(new['weights']-anchor),.03)
        for key in old:assert np.array_equal(old[key],new[key])
