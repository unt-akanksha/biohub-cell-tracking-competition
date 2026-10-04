import numpy as np
import pytest
from research.independent_motion_prior import link_motion


def test_obvious_tracks_and_no_coordinate_mutation():
    points = np.asarray([[0,1,1,1],[0,1,40,40],[1,1,2,1],[1,1,41,40]],float)
    original = points.copy()
    assert {(s,t) for s,t,_ in link_motion(points)} == {(0,2),(1,3)}
    assert np.array_equal(points,original)


def test_null_option_rejects_distant_only_parent():
    assert not link_motion([[0,0,0,0],[1,50,50,50]])


def test_equal_parent_tie_abstains_and_gap_is_not_bridged():
    assert not link_motion([[0,1,1,1],[0,1,1,1],[1,1,1,1]])
    assert not link_motion([[0,1,1,1],[2,1,1,1]])


def test_no_merges_and_at_most_two_daughters():
    edges = link_motion([[0,1,1,1],[1,1,1,1],[1,1,2,1],[1,1,1,2]])
    assert len(edges) == 2 and len({t for _,t,_ in edges}) == 2


def test_invalid_time_is_rejected():
    with pytest.raises(ValueError):
        link_motion([[.5,1,1,1]])
