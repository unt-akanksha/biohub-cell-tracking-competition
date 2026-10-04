from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from division_missing_null_loss import division_missing_null_loss
from sparse_division_balanced_loss import sparse_division_balanced_loss
from sparse_parent_missing_null import missing_null_targets
from sparse_parent_loss import sparse_parent_loss


def test_both_daughters_and_null_supervised_unknown_untouched():
    scores = torch.zeros(2,5,requires_grad=True)
    target = torch.tensor([[1.,1.,0.,0.,0.],[0.,0.,1.,0.,0.]])
    mask = [False,False,False,True,False]
    loss = division_missing_null_loss(scores,target,mask)
    augmented,_ = missing_null_targets(scores,target,mask)
    expected = .75*sparse_division_balanced_loss(scores,target) - .25*torch.log_softmax(augmented,0)[-1,3]
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert (scores.grad[0,:2] < 0).all() and scores.grad[1,2] < 0
    assert (scores.grad[:,3] > 0).all() and (scores.grad[:,4] == 0).all()


def test_without_null_is_exact_existing_balanced_loss():
    scores = torch.randn(2,3)
    target = torch.tensor([[1.,1.,0.],[0.,0.,1.]])
    torch.testing.assert_close(division_missing_null_loss(scores,target,[False]*3),
                               sparse_division_balanced_loss(scores,target))


def test_null_only_and_unannotated_only_are_well_defined():
    scores = torch.randn(2,3,requires_grad=True)
    target = torch.zeros_like(scores)
    torch.testing.assert_close(division_missing_null_loss(scores,target,[False,True,False]),
        sparse_parent_loss(*missing_null_targets(scores,target,[False,True,False])))
    empty = division_missing_null_loss(scores,target,[False]*3)
    empty.backward()
    assert empty.item() == 0 and (scores.grad == 0).all()


def test_contradictory_null_rejected():
    with pytest.raises(ValueError):
        division_missing_null_loss(torch.zeros(2,2),torch.eye(2),[True,False])
