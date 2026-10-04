import numpy as np
import pytest
from research.focus_presence_trees import fit,predict,SETTINGS


def test_role_guard_before_dependency_import():
    with pytest.raises(ValueError):fit({},'diagnostic')


def test_float32_threshold_boundary_and_joint_logit():
    threshold=float(np.float32(1.))+1e-8
    model=dict(settings=SETTINGS,initial_log_odds=2.,trees=[dict(left=[1,-1,-1],right=[2,-1,-1],feature=[0,-2,-2],threshold=[threshold,-2,-2],value=[0.,-3.,4.])])
    x=np.zeros((2,8));x[:,0]=[1.+2e-8,1.+2e-7]
    assert np.allclose(predict(model,x),[1.85,2.2])
