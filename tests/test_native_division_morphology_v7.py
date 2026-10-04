import numpy as np
import pytest
from research.native_division_morphology_v7 import features,patch_profiles,fit,predict,calibrate,metrics


def fixture():
    axis=np.arange(-7,8);r2=sum(a*a for a in np.meshgrid(axis,axis,axis,indexing='ij'))
    patches=np.stack([np.stack([np.exp(-r2/(2*s*s))]*3) for s in (2.,1.5,1.7)])
    return patches,np.array([[0,1,2]]),np.array([[[0.,0,0],[0,1,1],[0,-1,-1]]])


def test_daughter_exchange_invariance_and_finite_blank_images():
    p,t,c=fixture();old=features(p,t,c);new=features(p,t[:,[0,2,1]],c[:,[0,2,1]])
    for key in old:np.testing.assert_array_equal(old[key],new[key])
    assert old['morphology'].shape==(1,18)
    assert np.isfinite(patch_profiles(np.zeros_like(p))).all()


def test_profile_ratios_ignore_common_intensity_gain_and_background():
    p,t,c=fixture()
    np.testing.assert_allclose(features(p,t,c)['morphology'],features(p*3+7,t,c)['morphology'],rtol=1e-8,atol=1e-8)


def test_soft_volume_and_radius_distinguish_nucleus_width():
    p,_,_=fixture();profiles=patch_profiles(p)
    assert profiles[0,2]>profiles[2,2]>profiles[1,2]
    assert profiles[0,3]>profiles[2,3]>profiles[1,3]


def test_classifier_reload_and_no_false_positive_calibration(tmp_path):
    x=np.array([[-2.,1],[-1,1],[1,1],[2,1]]);y=np.array([0,0,1,1])
    model=fit(x,y);p=predict(model,x);result=metrics(y,p,calibrate(y,p))
    assert result['tp']==2 and result['fp']==0
    path=tmp_path/'model.npz';np.savez(path,**model)
    with np.load(path) as data:np.testing.assert_array_equal(p,predict(dict(data),x))
    with pytest.raises(ValueError):fit(x,np.zeros(4))
