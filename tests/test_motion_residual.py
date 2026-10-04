from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import numpy as np
import torch
from motion_residual import parent_probabilities,add_null_target,install_motion_residual
from independent_motion_prior import SCALE,VARIANCE,NULL_LOGIT


class ZeroModel:
    def predict_edges(self,*values):
        return torch.zeros(values[2].shape[0],values[2].shape[1],values[3].shape[1])


def test_inference_matches_frozen_numpy_prior():
    source = torch.tensor([[[1.,1.,1.],[1.,40.,40.]]])
    target = torch.tensor([[[1.,2.,1.],[1.,41.,40.]]])
    delta = (source.numpy()[:,:,None]-target.numpy()[:,None])*SCALE
    weights = np.exp(-.5*(delta**2/VARIANCE).sum(-1))
    expected = weights/(weights.sum(1,keepdims=True)+np.exp(NULL_LOGIT))
    model = ZeroModel()
    install_motion_residual(model,inference=True)
    actual = model.predict_edges(None,None,source,target).sigmoid().numpy()
    assert np.allclose(actual,expected,atol=2e-7)


def test_null_is_not_a_real_node_and_receives_no_positive_target():
    scores = torch.zeros(3,2,requires_grad=True)
    target = torch.zeros_like(scores); target[1,0] = 1
    extended,labels = add_null_target(scores,target)
    assert extended.shape == labels.shape == (4,2)
    assert not labels[-1].any()
    probs = parent_probabilities(scores)
    assert probs.shape == scores.shape and torch.all(probs.sum(0) < 1)
    (-torch.log(probs[1,0])).backward()
    assert torch.isfinite(scores.grad).all()


def test_far_single_parent_can_be_unassigned():
    assert parent_probabilities(torch.tensor([[-50.]])).item() < .5


def test_training_uses_same_spatial_prior_as_inference():
    source = torch.tensor([[[1.,1.,1.]]]); target = torch.tensor([[[2.,3.,4.]]])
    training,inference = ZeroModel(),ZeroModel()
    install_motion_residual(training)
    install_motion_residual(inference,inference=True)
    raw = training.predict_edges(None,None,source,target)
    assert torch.allclose(parent_probabilities(raw),inference.predict_edges(None,None,source,target).sigmoid())
