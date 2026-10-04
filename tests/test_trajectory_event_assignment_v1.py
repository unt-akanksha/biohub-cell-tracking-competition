from types import SimpleNamespace
import numpy as np
import pytest

import research.trajectory_event_assignment_v1 as event


def tiny():
    # death, two continuations, one fork, two births
    return event.problem([(0,-1,-1), (0,0,-1), (0,1,-1), (0,0,1), (-1,0,-1), (-1,1,-1)], 1, 2)


def test_two_known_daughters_are_feasible_not_discarded_as_duplicate_parents():
    case, reason = event.prepare(tiny(), np.eye(6), np.array([0,0]), np.zeros((6,2), bool))
    assert reason == 'prepared' and case['n_constraints'] == 2
    chosen = event.solve(case['problem'], np.zeros(6), allowed=case['allowed'])
    assert chosen.tolist() == [0,0,0,1,0,0]


def test_competing_divisions_cannot_claim_same_daughter():
    case = event.problem([(0,-1,-1), (1,-1,-1), (0,0,1), (1,1,2),
                          (-1,0,-1), (-1,1,-1), (-1,2,-1)], 2, 3)
    chosen = event.solve(case, [0,0,5,4,0,0,0])
    assert chosen[2] == 1 and chosen[3] == 0
    event.validate_choice(case, chosen)


def test_unknown_child_does_not_become_negative_division_evidence():
    case, _ = event.prepare(tiny(), np.eye(6), np.array([0,-1]), np.ones((6,2), bool))
    assert case['allowed'][3] and case['margin'][3] == 0
    assert case['allowed'][5] and case['margin'][5] == 0
    assert not case['allowed'][4] and case['margin'][4] == 1


def test_close_wrong_parent_is_latently_excluded_but_not_given_negative_margin():
    problem = event.problem([(0,-1,-1), (1,-1,-1), (0,0,-1), (1,0,-1), (-1,0,-1)], 2, 1)
    case, _ = event.prepare(problem, np.eye(5), np.array([0]), np.zeros((5,2), bool))
    assert not case['allowed'][3] and case['margin'][3] == 0
    assert case['margin'][4] == 1


def test_missing_parent_is_not_assumed_to_be_a_birth_label():
    with pytest.raises(ValueError, match='never assumed births'):
        event.prepare(tiny(), np.eye(6), np.array([0,-2]), np.zeros((6,2), bool))


def test_partial_hinge_gradient_matches_finite_difference_away_from_ties():
    case, _ = event.prepare(tiny(), np.eye(6), np.array([0,0]), np.zeros((6,2), bool))
    weights = np.array([.1,.8,.2,-.5,-.1,-.2])
    value, gradient = event.hinge(case, weights)
    finite = np.empty(6)
    for i in range(6):
        step = np.zeros(6); step[i] = 1e-5
        finite[i] = (event.hinge(case, weights+step)[0] - event.hinge(case, weights-step)[0]) / 2e-5
    assert value > 0
    assert np.allclose(gradient, finite, atol=1e-7)


def test_timeout_falls_back_for_inference_but_is_not_a_training_oracle(monkeypatch):
    problem = tiny()
    case, _ = event.prepare(problem, np.eye(6), np.array([0,0]), np.zeros((6,2), bool))
    incumbent = np.array([0,1,0,0,0,1])
    monkeypatch.setattr(event, 'milp', lambda *a, **kw: SimpleNamespace(status=1, success=False, x=np.ones(6)))
    chosen, report = event.infer(problem, np.zeros(6), incumbent)
    assert np.array_equal(chosen, incumbent) and report['fallback']
    with pytest.raises(event.EventSolveError):
        event.hinge(case, np.zeros(6))
    with pytest.raises(event.EventSolveError):
        event.prepare(problem, np.eye(6), np.array([0,0]), np.zeros((6,2), bool))


def test_invalid_and_duplicate_event_definitions_are_rejected():
    for options in [[(0,0,0)], [(-1,0,1)], [(0,-1,1)], [(0,1,0)], [(0,0,-1),(0,0,-1)]]:
        with pytest.raises(ValueError):
            event.problem(options, 1, 2)


def test_zero_gain_preserves_incumbent_exactly():
    incumbent = np.array([0,1,0,0,0,1])
    chosen, report = event.infer(tiny(), np.zeros(6), incumbent)
    assert np.array_equal(chosen, incumbent) and not report['changed']
