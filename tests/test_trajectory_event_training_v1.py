import numpy as np
import pytest

from research.trajectory_event_assignment_v1 import problem, prepare, hinge, EventSolveError
import research.trajectory_event_training_v1 as training


def case():
    p = problem([(0,-1,-1), (0,0,-1), (0,1,-1), (0,0,1), (-1,0,-1), (-1,1,-1)], 1, 2)
    return prepare(p, np.eye(6), np.array([0,0]), np.zeros((6,2),bool))[0]


def test_real_exact_oracle_fit_is_deterministic_and_learns_known_fork():
    source = case()
    anchor = np.array([.1,.8,.2,-.5,-.1,-.2])
    visits = []
    weights, history = training.fit([source], anchor, np.full(6,.01), epochs=8,
                                   learning_rate=.1, callback=lambda w,r: visits.append(r))
    repeated, _ = training.fit([source], anchor, np.full(6,.01), epochs=8, learning_rate=.1)
    assert np.array_equal(weights,repeated) and len(history)==len(visits)==8
    assert hinge(source,weights)[0] < hinge(source,anchor)[0]
    assert weights[3] > anchor[3]


def test_training_timeout_is_not_suppressed(monkeypatch):
    source = case()
    def fail(*a, **kw):
        raise EventSolveError('timeout',status=1)
    monkeypatch.setattr(training,'hinge',fail)
    with pytest.raises(EventSolveError):
        training.fit([source],np.zeros(6),np.ones(6))


def test_event_prior_is_finite_and_has_source_provenance_not_fixed_pilot_cost():
    edge = np.arange(18,dtype=float)/18
    anchor = training.prior(edge,12,6444)
    assert np.array_equal(anchor[:18],edge)
    assert anchor[19] == pytest.approx(np.log(12.5/6432.5))
    assert np.isfinite(training.prior(edge,0,100)).all()
    with pytest.raises(ValueError):
        training.prior(edge,101,100)
