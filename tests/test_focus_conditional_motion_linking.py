import numpy as np
from research.focus_conditional_motion import FEATURES
from research.focus_conditional_motion_linking import link
from research.focus_residual_calibration import link as original


def model(mean):
    coefficients=np.zeros((7,3));coefficients[0]=mean
    return dict(features=list(FEATURES),ridge=1.,center=[0.]*6,scale=[1.]*6,coefficients=coefficients.tolist(),variance_um2=[2.,2.,2.])


def test_constant_conditional_model_exactly_replays_original_gaussian():
    coords=np.array([[0,1,2,3],[0,3,9,9],[1,1,2,4],[1,3,9,9]],float)
    flow=np.array([[0,0,0],[0,0,0],[.2,.1,-.2],[.1,.3,-.1]],np.float32)
    oldcoords=coords.copy();oldflow=flow.copy();mean=[.2,-.1,.3]
    # Floating arithmetic can differ in the last probability bit; graph identities
    # and posterior values must agree to machine precision.
    actual=link(coords,flow,model(mean));expected=original(coords,flow,dict(mean_um=mean,variance_um2=[2.,2.,2.]))
    assert [(s,d) for s,d,p in actual]==[(s,d) for s,d,p in expected]
    np.testing.assert_allclose([p for s,d,p in actual],[p for s,d,p in expected],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(coords,oldcoords);np.testing.assert_array_equal(flow,oldflow)


def test_zero_correction_exact_replay():
    coords=np.array([[0,2,2,2],[1,2,2,2]],float);flow=np.zeros((2,3),np.float32)
    assert link(coords,flow,model([0,0,0]))==original(coords,flow,dict(mean_um=[0,0,0],variance_um2=[2.,2.,2.]))
