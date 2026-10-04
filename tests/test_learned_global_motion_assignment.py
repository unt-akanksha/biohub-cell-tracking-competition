from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from learned_global_motion_assignment import link
from global_motion_assignment import link as static_link


def test_zero_motion_exactly_replays_static_assignment():
    coords=np.array([[0,2,5,5],[0,2,8,8],[1,2,6,5],[1,2,9,8],[3,2,5,5]],float)
    assert link(coords,np.zeros((len(coords),3)))==static_link(coords)


def test_backward_direction_uses_child_flow_in_physical_units():
    coords=np.array([[0,0,0,0],[1,0,0,20]],float)
    flow=np.array([[0,0,0],[0,0,-20*.40625]])
    original=coords.copy(); saved=flow.copy()
    assert static_link(coords)==[]
    assert link(coords,flow)==[(0,1)]
    assert link(coords,-flow)==[]
    np.testing.assert_array_equal(coords,original); np.testing.assert_array_equal(flow,saved)


def test_flow_alignment_and_frame_zero_guard():
    coords=np.array([[0,0,0,0],[1,0,0,0]],float)
    for flow in (np.zeros((1,3)),np.ones((2,3)),np.full((2,3),np.nan)):
        with pytest.raises(ValueError): link(coords,flow)
