import json
import numpy as np
import pytest
from research.focus_conditional_motion import examples,fit,predict,metrics,heldout_gate
from research.focus_residual_calibration import matched_residuals


def test_targets_replay_original_and_exclude_partial_division():
    coords=np.array([[0,2,3,4],[1,3,4,5],[0,5,5,5],[1,6,5,5]],float);flow=np.zeros((4,3))
    mapping={0:10,1:11,2:20,3:21};edges=[(10,11),(20,21),(20,22)]
    result=examples(coords,flow,mapping,edges)
    np.testing.assert_array_equal(result['y'],matched_residuals(coords,flow,mapping,edges))
    np.testing.assert_array_equal(result['pairs'],[[0,1]])


def test_ridge_learns_known_conditional_relation_and_serialization():
    rng=np.random.default_rng(123);x=rng.normal(size=(1000,6));y=x[:,:3]*2+1
    model=fit(x,y);restored=json.loads(json.dumps(model))
    np.testing.assert_array_equal(predict(model,x),predict(restored,x))
    assert np.mean((predict(model,x)-y)**2)<1e-4
    assert model['observations']==1000


def test_training_standardization_not_changed_by_prediction():
    rng=np.random.default_rng(1);x=rng.normal(size=(100,6));y=rng.normal(size=(100,3))
    model=fit(x,y);before=json.dumps(model);predict(model,np.full((3,6),1000.))
    assert json.dumps(model)==before


def test_gaussian_nll_and_invalid_shapes():
    row=metrics(np.zeros((3,3)),np.zeros(3),np.ones(3))
    assert row['mse_um2']==0 and row['nll']==pytest.approx(1.5*np.log(2*np.pi))
    with pytest.raises(ValueError):fit(np.zeros((100,5)),np.zeros((100,3)))


def test_gate_cannot_pass_only_pooled_gain_without_movie_majority():
    rows=[]
    for i in range(14):
        b=metrics(np.ones((100,3)),np.zeros(3),np.ones(3))
        c=metrics(np.ones((100,3)),np.ones(3) if i<6 else np.full(3,-.01),np.ones(3))
        rows.append(dict(stem=str(i),baseline=b,candidate=c))
    gate=heldout_gate(rows)
    assert gate['conditions']['pooled_mse_improves']
    assert not gate['conditions']['majority_movies_nll_improve'] and not gate['passed']
