import numpy as np
import pytest
from research.registration_flow_anchor import anchor


def test_forward_sign_and_local_differences():
    flow = np.array([[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]])
    out = anchor(flow, [4.,-2.,1.], 1.)
    np.testing.assert_allclose(np.median(out,axis=0),[-4.,2.,-1.])
    np.testing.assert_allclose(out[1:]-out[:-1],flow[1:]-flow[:-1])
    np.testing.assert_array_equal(flow,[[1,2,3],[2,3,4],[3,4,5]])


def test_neutral_uncertain_and_empty():
    flow = np.ones((3,3))
    np.testing.assert_array_equal(anchor(flow,[1,2,3],0),flow)
    np.testing.assert_allclose(anchor(flow,[1,2,3],.5),[[0,-.5,-1]]*3)
    assert anchor(np.empty((0,3)),[0,0,0],1).shape == (0,3)


@pytest.mark.parametrize('weight',[float('nan'),float('inf'),-1,1.1])
def test_invalid_weights(weight):
    with pytest.raises(ValueError): anchor(np.zeros((2,3)),[0,0,0],weight)


def test_nonfinite_and_shapes():
    with pytest.raises(ValueError): anchor([[float('nan')]*3],[0,0,0],.5)
    with pytest.raises(ValueError): anchor(np.zeros((2,2)),[0,0,0],.5)
