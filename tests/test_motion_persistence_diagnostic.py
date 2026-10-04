from pathlib import Path
import runpy

import numpy as np
import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/analyze-independent-motion-persistence.py'))


def test_known_causal_shrinkage_recovered():
    x = np.arange(30,dtype=float).reshape(10,3)-10
    y = x*np.array([.2,.5,.8])+np.array([1.,-2.,3.])
    model = MODULE['fit'](x,y)
    np.testing.assert_allclose(model['alpha'],[.2,.5,.8])
    np.testing.assert_allclose(model['intercept_um'],[1.,-2.,3.])
    assert MODULE['error'](x*np.asarray(model['alpha'])+model['intercept_um'],y)['rmse_per_coordinate_um'] < 1e-12


def test_reversal_and_amplification_are_not_fitted():
    x = np.arange(30,dtype=float).reshape(10,3)
    model = MODULE['fit'](x,x*np.array([-1.,2.,.5]))
    np.testing.assert_allclose(model['alpha'],[0.,1.,.5])


def test_nonfinite_input_rejected():
    with pytest.raises(ValueError):
        MODULE['fit']([[0,0,0],[1,1,float('nan')]],np.zeros((2,3)))
