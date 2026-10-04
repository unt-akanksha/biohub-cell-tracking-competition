import numpy as np
import pytest
from research.calibrated_motion_scores import calibrated_motion_scores


def test_zero_weight_preserves_finite_scores_without_evaluating_neural():
    generator=np.random.default_rng(944)
    prior=-generator.uniform(0,100,(2,512,300)).astype(np.float32)
    neural=generator.normal(size=prior.shape).astype(np.float32)
    def forbidden(): raise AssertionError('Zero-weight network must not execute')
    for beta in (.25,1.,4.):
        expected=calibrated_motion_scores(prior,lambda:neural,0.,beta)
        actual=calibrated_motion_scores(prior,forbidden,0.,beta,skip_zero_neural=True)
        np.testing.assert_array_equal(actual,expected)


def test_nonzero_weights_and_default_still_execute_network():
    calls=[]
    def neural(): calls.append(True); return np.array([2.,3.])
    prior=np.array([-1.,-2.])
    np.testing.assert_array_equal(calibrated_motion_scores(prior,neural,.3,1.,skip_zero_neural=True),.3*neural()+prior)
    calibrated_motion_scores(prior,neural,0.,1.)
    assert len(calls)==3
    with pytest.raises(ValueError): calibrated_motion_scores(prior,neural,0.,1.,skip_zero_neural=1)
