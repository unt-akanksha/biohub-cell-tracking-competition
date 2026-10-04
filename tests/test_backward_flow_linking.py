from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.backward_flow_linking import link_backward_flow
from research.independent_motion_prior import link_motion


def test_zero_flow_equals_existing_static_control_exactly():
    rng = np.random.default_rng(18)
    points = np.column_stack([np.repeat(np.arange(4),20),rng.integers(0,40,(80,3))])
    assert link_backward_flow(points,np.zeros((80,3))) == link_motion(points)


def test_backward_sign_resolves_two_equidistant_parents():
    points = np.array([[0,2,2,0],[0,2,2,8],[1,2,2,4]],float)
    assert link_motion(points) == []
    flow = np.zeros((3,3)); flow[2,2] = 1.625
    assert [(s,d) for s,d,_ in link_backward_flow(points,flow)] == [(1,2)]
    flow[2,2] = -1.625
    assert [(s,d) for s,d,_ in link_backward_flow(points,flow)] == [(0,2)]


def test_daughters_share_parent_but_third_child_is_not_allowed():
    points = np.array([[0,2,2,4],[1,2,2,2],[1,2,2,6],[1,2,2,4]],float)
    flow = np.zeros((4,3)); flow[1,2] = .8125; flow[2,2] = -.8125
    edges = link_backward_flow(points,flow)
    assert [(s,d) for s,d,_ in edges] == [(0,1),(0,2)]
    np.testing.assert_array_equal(points[:,3],[4,2,6,4])


def test_empty_gaps_nonfinite_and_misalignment():
    assert link_backward_flow(np.empty((0,4)),np.empty((0,3))) == []
    assert link_backward_flow([[0,1,1,1],[2,1,1,1]],np.zeros((2,3))) == []
    for bad in (np.full((2,3),np.nan),np.zeros((1,3))):
        with pytest.raises(ValueError):
            link_backward_flow([[0,1,1,1],[1,1,1,1]],bad)
