import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('frozen_head_test', Path(__file__).parents[1]/'research/frozen_image_head.py')
head = importlib.util.module_from_spec(spec); spec.loader.exec_module(head)


def test_morphology_daughter_symmetry_and_empty_context():
    c = np.zeros((2,43,8)); m = np.zeros((2,43), dtype=bool); m[:,:3] = True
    c[:,1,1:4] = [.2,.1,0]; c[:,2,1:4] = [-.1,.2,0]
    c[0,3] = [-.5,.1,.2,0,.4,0,0,1.5]; m[0,3] = True
    before = head.morphology(c,m)
    c[:,[1,2]] = c[:,[2,1]]
    np.testing.assert_array_equal(before, head.morphology(c,m))
    assert before.shape == (2,45) and np.isfinite(before).all()


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(12)
    x=rng.normal(size=(13,4)); y=(np.arange(13)%3==0).astype(float)
    weights=np.ones(13)/13; theta=rng.normal(size=5)
    loss, grad=head.objective(theta,x,y,weights,.1)
    for k in range(5):
        d=np.eye(5)[k]*1e-6
        actual=(head.objective(theta+d,x,y,weights,.1)[0]-head.objective(theta-d,x,y,weights,.1)[0])/2e-6
        np.testing.assert_allclose(actual,grad[k],rtol=1e-6,atol=1e-8)
    assert np.isfinite(loss)


def test_source_transform_is_frozen_and_serializable(tmp_path):
    rng=np.random.default_rng(41)
    x=rng.normal(size=(24,sum(head.BLOCKS)))
    y=(np.arange(24)%2).astype(float); x[:,0]=y*4-2
    state=head.fit(x,y,np.ones(24,dtype=bool),np.ones(24),.01)
    score=head.predict(x,state); saved=tmp_path/'head.npz'
    np.savez(saved,**state)
    with np.load(saved,allow_pickle=False) as a: replay=head.predict(x,dict(a))
    np.testing.assert_array_equal(score,replay)
    mean=state['mean'].copy(); head.predict(x+1000,state)
    np.testing.assert_array_equal(mean,state['mean'])
    assert score[y==1].mean() > score[y==0].mean()


def test_weights_equalize_present_class_eligibility_strata():
    y=np.array([0,0,0,1,1]); e=np.array([0,0,1,0,1])
    weights=head.stratum_weights(y,e,np.array([1,2,3,4,5]))
    for a in (0,1):
        for b in (0,1):
            assert np.isclose(weights[(y==a)&(e==b)].sum(),.25)
