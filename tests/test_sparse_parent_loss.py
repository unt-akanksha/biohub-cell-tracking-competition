"""Run in a functioning PyTorch environment before the GPU smoke launch."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location('sparse_parent_loss',
    Path(__file__).resolve().parents[1]/'research/sparse_parent_loss.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loss = module.sparse_parent_loss


def test_uniform_parent_loss_is_log_candidate_count_not_inverse_count():
    import math
    for n in (2,100,500):
        logits = torch.zeros(n,2,requires_grad=True)
        target = torch.zeros_like(logits)
        target[0,0] = 1
        value = loss(logits,target)
        assert value.item() == pytest.approx(math.log(n))
        value.backward()
        assert logits.grad[0,0].item() == pytest.approx(-(1-1/n))
        assert torch.equal(logits.grad[:,1],torch.zeros(n))


def test_division_children_can_share_parent():
    logits = torch.tensor([[8.,8.],[-8.,-8.]],requires_grad=True)
    target = torch.tensor([[1.,1.],[0.,0.]])
    assert loss(logits,target).item() < 1e-5


def test_unannotated_children_do_not_change_loss():
    logits = torch.tensor([[1.,999.],[0.,-999.]],requires_grad=True)
    target = torch.tensor([[1.,0.],[0.,0.]])
    assert torch.equal(loss(logits,target),loss(logits[:,:1],target[:,:1]))


def test_empty_supervision_has_zero_finite_gradient():
    logits = torch.randn(3,4,requires_grad=True)
    loss(logits,torch.zeros_like(logits)).backward()
    assert torch.equal(logits.grad,torch.zeros_like(logits))


def test_merges_are_rejected():
    with pytest.raises(ValueError,match='Multiple'):
        loss(torch.zeros(2,1),torch.ones(2,1))
