import numpy as np
import pytest
from scipy.special import softmax
from research.focus_parent_presence import FEATURES,summarize
from research.focus_presence_inference import context,posterior,edges


def example(ns=3,nt=5):
    rng=np.random.default_rng(433)
    packet=dict(source_features=rng.normal(size=(ns,32)),target_features=rng.normal(size=(nt,32)),
        source_coords=rng.uniform(0,10,(ns,3)),labels=np.arange(nt)%ns,target_indices=np.arange(ns,ns+nt,dtype=np.int64))
    return packet,rng.normal(size=(ns,nt)),rng.normal(size=(ns,nt))


@pytest.mark.parametrize('ns',[1,3,20])
def test_label_free_context_exactly_replays_training_summary(ns):
    packet,scores,prior=example(ns)
    expected=summarize(packet,scores,prior)['context']
    packet.pop('labels');packet.pop('target_indices')
    assert np.array_equal(context(packet,scores,prior)[0],expected)


def test_zero_correction_replays_original_joint_softmax():
    packet,scores,prior=example();real,null=posterior(packet,scores,prior)
    expected=softmax(np.vstack([scores,np.full(scores.shape[1],-4.5)]),axis=0)
    np.testing.assert_allclose(real,expected[:-1],rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(null,expected[-1],rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(real.sum(0)+null,1,atol=1e-15)


def test_presence_correction_preserves_conditional_parent_ranking():
    packet,scores,prior=example();base,_=posterior(packet,scores,prior)
    model=dict(features=list(FEATURES),role='fitting',mean=[0.]*7,std=[1.]*7,theta=[-2.]+[0.]*7)
    corrected,null=posterior(packet,scores,prior,model)
    assert np.array_equal(base.argmax(0),corrected.argmax(0))
    np.testing.assert_allclose(corrected/corrected.sum(0),base/base.sum(0),rtol=1e-14)
    assert (corrected.sum(0)<base.sum(0)).all()


def test_graph_policy_strict_threshold_and_max_two_children():
    source=np.array([0,1],np.int64);target=np.array([2,3,4,5],np.int64)
    probabilities=np.array([[.8,.9,.7,0],[0,0,0,.5]])
    assert [(s,t) for s,t,_ in edges(source,target,probabilities)]==[(0,3),(0,2)]


def test_invalid_posterior_or_identity_rejected():
    with pytest.raises(ValueError):edges(np.array([0],np.int64),np.array([0],np.int64),np.array([[.9]]))
    with pytest.raises(ValueError):edges(np.array([0,1],np.int64),np.array([2],np.int64),np.array([[.8],[.8]]))
