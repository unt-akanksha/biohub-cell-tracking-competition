import numpy as np
import pytest
from research.trajectory_correction_logistic_v1 import fit,predict


def test_regularized_toy_fit_and_frozen_threshold():
    x=np.array([[-3,1],[-2,1],[-1,1],[1,1],[2,1],[3,1]],float);y=np.array([0,0,0,1,1,1])
    model=fit(x,y)
    assert model['threshold']==.5
    np.testing.assert_array_equal(predict(x,model)>=.5,y.astype(bool))
    assert np.isfinite(predict(np.array([[1e6,1]]),model)).all()


def test_invalid_training_and_model_rejected():
    with pytest.raises(ValueError):fit(np.zeros((4,2)),np.ones(4))
    with pytest.raises(ValueError):fit(np.full((4,2),np.nan),np.array([0,0,1,1]))
    model=fit(np.arange(8).reshape(4,2),np.array([0,0,1,1]));model['scale'][0]=0
    with pytest.raises(ValueError):predict(np.zeros((1,2)),model)
