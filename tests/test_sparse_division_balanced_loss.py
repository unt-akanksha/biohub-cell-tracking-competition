from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from motion_residual import add_null_target
from sparse_parent_loss import sparse_parent_loss
from sparse_division_balanced_loss import sparse_division_balanced_loss


def test_mixed_batch_uses_equal_class_means_and_keeps_both_daughters():
    scores = torch.tensor([[-12.,-14.,-4.,-3.,-3.], [-6.,-6.,3.,-3.,-3.],
                           [-6.,-6.,-4.,3.,-3.], [-6.,-6.,-4.,-3.,3.]], requires_grad=True)
    target = torch.tensor([[1.,1.,0.,0.,0.], [0.,0.,1.,0.,0.],
                           [0.,0.,0.,1.,0.], [0.,0.,0.,0.,1.]])
    augmented, truth = add_null_target(scores, target)
    nll = -(torch.log_softmax(augmented,0)*truth).sum(0)
    loss = sparse_division_balanced_loss(scores,target)
    torch.testing.assert_close(loss, .5*(nll[:2].mean()+nll[2:].mean()))
    loss.backward()
    assert scores.grad[0,0] < 0 and scores.grad[0,1] < 0
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize('target', [torch.eye(3), torch.tensor([[1.,1.,0.],[0.,0.,0.]])])
def test_single_class_is_exact_original_loss(target):
    scores = torch.randn_like(target, requires_grad=True)
    torch.testing.assert_close(sparse_division_balanced_loss(scores,target),
                               sparse_parent_loss(*add_null_target(scores,target)))


def test_unannotated_columns_receive_no_gradient_and_no_annotations_zero():
    scores = torch.randn(2,4,requires_grad=True)
    target = torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,0.]])
    sparse_division_balanced_loss(scores,target).backward()
    assert not scores.grad[:,3].any()
    scores.grad = None
    loss = sparse_division_balanced_loss(scores,torch.zeros_like(scores))
    loss.backward()
    assert loss.item() == 0 and not scores.grad.any()


def test_invalid_three_daughter_event_rejected():
    with pytest.raises(ValueError, match='more than two'):
        sparse_division_balanced_loss(torch.zeros(1,3),torch.ones(1,3))
