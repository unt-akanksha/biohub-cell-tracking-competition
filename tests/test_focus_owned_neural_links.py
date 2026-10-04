import numpy as np
import pytest
from research.focus_owned_neural_links import link
from research.backward_flow_linking import link_backward_flow


def test_zero_neural_replays_physical_graph_and_keeps_input():
    coords=np.array([[0,3,20,20],[0,3,40,40],[1,3,20,20],[1,3,40,40]],float)
    flow=np.zeros((4,3)); before=coords.copy()
    result=link(coords,flow,{0:np.zeros((2,2))})
    expected=link_backward_flow(coords,flow)
    assert [(s,d) for s,d,p in result]==[(s,d) for s,d,p in expected]
    np.testing.assert_allclose([p for s,d,p in result],[p for s,d,p in expected],rtol=1e-14)
    np.testing.assert_array_equal(coords,before)


def test_null_stability_and_degree_limits():
    coords=np.array([[0,1,1,1],[1,1,1,1],[1,1,2,1],[1,1,3,1]],float)
    assert not link(coords,np.zeros((4,3)),{0:np.full((1,3),-10000.)})
    edges=link(coords,np.zeros((4,3)),{0:np.full((1,3),10000.)})
    assert len(edges)==2 and all(np.isfinite(e[2]) for e in edges)


def test_shapes_nonfinite_and_unexpected_pairs_fail():
    coords=np.array([[0,1,1,1],[1,1,1,1]],float);flow=np.zeros((2,3))
    for scores in ({0:np.ones((2,2))},{0:np.full((1,1),np.nan)},{0:np.zeros((1,1)),1:np.zeros((1,1))}):
        with pytest.raises(ValueError):link(coords,flow,scores)
