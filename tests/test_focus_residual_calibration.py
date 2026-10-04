import numpy as np
import pytest
from research.focus_residual_calibration import fit,link,matched_residuals,VARIANCE_FLOOR
from research.independent_motion_prior import VARIANCE
from research.backward_flow_linking import link_backward_flow


def test_mle_mean_variance_and_floor():
    values=np.zeros((100,3));values[:,0]=np.tile([-2.,2.],50);values[:,1]=3.
    result=fit(values)
    np.testing.assert_array_equal(result['mean_um'],[0,3,0])
    np.testing.assert_allclose(result['variance_um2'],[4,VARIANCE_FLOOR[1],VARIANCE_FLOOR[2]])
    with pytest.raises(ValueError): fit(values[:99])


def test_original_parameters_exactly_replay_linking_without_node_edits():
    coords=np.array([[0,1,2,3],[0,1,40,40],[1,1,2,3],[1,1,3,3],[1,1,40,40]],np.float32)
    flow=np.zeros((len(coords),3),np.float32);original=coords.copy()
    assert link(coords,flow,dict(mean_um=[0,0,0],variance_um2=VARIANCE.tolist()))==link_backward_flow(coords,flow)
    np.testing.assert_array_equal(coords,original)


def test_training_residual_sign_and_divisions_excluded():
    coords=np.array([[0,5,2,3],[1,3,2,3],[0,1,40,40],[1,1,40,40],[1,1,41,40]],float)
    mapping={i:i+10 for i in range(5)}
    residual=matched_residuals(coords,np.zeros((5,3)),mapping,[(10,11),(12,13),(12,14)])
    np.testing.assert_allclose(residual,[[3.25,0,0]])
    parameters=dict(mean_um=[3.25,0,0],variance_um2=VARIANCE_FLOOR.tolist())
    assert any((s,d)==(0,1) for s,d,p in link(coords,np.zeros((5,3)),parameters))


def test_bad_parameters_and_ambiguous_training_matches_rejected_or_excluded():
    coords=np.array([[0,1,1,1],[0,1,1,1],[1,1,1,1]],float)
    assert len(matched_residuals(coords,np.zeros((3,3)),{0:10,1:10,2:11},[(10,11)]))==0
    with pytest.raises(ValueError): link(coords,np.zeros((3,3)),dict(mean_um=[0,0,0],variance_um2=[0,0,0]))
