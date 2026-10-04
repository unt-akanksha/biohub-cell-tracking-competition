import numpy as np
import pytest

from research.division_preserving_flow_assignment import ordinary_assignment, link, forks, validate_edges
from research.backward_flow_linking import link_backward_flow


def test_joint_assignment_and_private_nulls():
    assert ordinary_assignment([[1, 1.1], [1.2, 4]]) == [(0, 1), (1, 0)]
    assert ordinary_assignment([[0, 4.5, 9]]) == [(0, 0)]
    assert ordinary_assignment(np.empty((0, 2))) == []
    assert ordinary_assignment(np.empty((2, 0))) == []


def test_fork_exactly_preserved_with_ordinary_links_and_no_node_changes():
    coords = np.array([[0, 2, 5, 5], [0, 2, 80, 80],
                       [1, 2, 5, 5], [1, 2, 5, 6], [1, 2, 80, 80],
                       [3, 2, 80, 80]], dtype=np.float32)
    flow = np.zeros((len(coords), 3), dtype=np.float32)
    copy = coords.copy()
    edges, receipt = link(coords, flow)
    assert edges == [(0, 2), (0, 3), (1, 4)]
    assert receipt['locked_divisions'] == 1
    assert forks(edges) == forks([(s, d) for s, d, _ in link_backward_flow(coords, flow)])
    np.testing.assert_array_equal(coords, copy)


def test_flow_is_used_and_first_frame_flow_must_be_zero():
    coords = np.array([[0, 1, 1, 1], [1, 1, 100, 100]], dtype=np.float32)
    flow = np.array([[0, 0, 0], [0, -99*.40625, -99*.40625]])
    assert link(coords, flow)[0] == [(0, 1)]
    flow[0, 0] = 1
    with pytest.raises(ValueError):
        link(coords, flow)


def test_empty_and_missing_frames():
    assert link(np.empty((0, 4)), np.empty((0, 3)))[0] == []
    assert link(np.array([[1, 1, 1, 1], [3, 1, 1, 1]]), np.zeros((2, 3)))[0] == []


@pytest.mark.parametrize('costs', [[[np.nan]], [[np.inf]], [[-1]], [1, 2]])
def test_bad_costs_rejected(costs):
    with pytest.raises(ValueError):
        ordinary_assignment(costs)


def test_guard_and_topology_no_silent_repair():
    with pytest.raises(ValueError, match='guard'):
        ordinary_assignment(np.zeros((2049, 1)))
    points = np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 2]])
    with pytest.raises(ValueError, match='division'):
        validate_edges(points, [(0, 1)], {0: (1, 2)})
    with pytest.raises(ValueError, match='division'):
        validate_edges(points, [(0, 1), (0, 2)], {})
    with pytest.raises(ValueError, match='Duplicate'):
        validate_edges(points, [(0, 1), (0, 1)], {})
