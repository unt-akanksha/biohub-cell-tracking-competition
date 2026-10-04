from itertools import product
import numpy as np
import pytest
from research.trajectory_event_assignment_v1 import problem
from research.trajectory_event_null_dominance_v1 import allowed_options


def simple():
    return problem(np.array([[0, -1, -1], [-1, 0, -1], [0, 0, -1]]), 1, 1)


def test_continuation_dominated_by_allowed_death_and_birth():
    mask, report = allowed_options(simple(), np.array([0., 0., -1.]))
    assert mask.tolist() == [True, True, False]
    assert report['null_dominated_continuations'] == 1


@pytest.mark.parametrize('mask', [[False, True, True], [True, False, True]])
def test_unavailable_replacement_cannot_prune_required_link(mask):
    result, _ = allowed_options(simple(), np.array([0., 0., -1.]), np.array(mask))
    assert result.tolist() == mask


def test_ties_and_near_ties_preserved():
    for value in (0., -1e-12):
        mask, _ = allowed_options(simple(), np.array([0., 0., value]))
        assert mask.all()


def test_fork_dominated_by_death_and_two_births_without_links():
    case = problem(np.array([[0, -1, -1], [-1, 0, -1], [-1, 1, -1], [0, 0, 1]]), 1, 2)
    mask, report = allowed_options(case, np.array([0., 0., 0., -1.]))
    assert mask.tolist() == [True, True, True, False]
    assert report['null_dominated_forks'] == 1


def test_all_optimal_assignments_survive_exhaustive_small_problems():
    options = [[-1, c, -1] for c in range(2)]
    for p in range(2):
        options += [[p, -1, -1], [p, 0, -1], [p, 1, -1], [p, 0, 1]]
    case = problem(np.array(options), 2, 2)
    choices = np.array(list(product((0, 1), repeat=len(options))))
    feasible = np.asarray(case['matrix'] @ choices.T == 1).all(axis=0)
    choices = choices[feasible]
    rng = np.random.default_rng(20260914)
    for _ in range(30):
        scores = rng.integers(-5, 6, size=len(options)).astype(float)
        allowed = rng.random(len(options)) > .2
        permitted = choices[np.all(choices[:, ~allowed] == 0, axis=1)]
        retained, _ = allowed_options(case, scores, allowed)
        assert not (retained & ~allowed).any()
        if len(permitted):
            values = permitted @ scores
            optimal = permitted[values == values.max()]
            assert (optimal[:, ~retained] == 0).all()
