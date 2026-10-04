from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
import torch
from sparse_parent_row_loss import sparse_parent_row_loss


def test_unannotated_child_false_fork_is_penalized_for_known_parent():
    scores = torch.tensor([[8.,8.],[-8.,-8.]],requires_grad=True)
    target = torch.tensor([[1.,0.],[0.,0.]])
    loss = sparse_parent_row_loss(scores,target)
    assert loss.item() > 8
    loss.backward()
    assert scores.grad[0,1] > 0 and torch.isfinite(scores.grad).all()


def test_true_annotated_division_is_not_suppressed():
    scores = torch.tensor([[8.,8.],[-8.,-8.]],requires_grad=True)
    target = torch.tensor([[1.,1.],[0.,0.]])
    assert sparse_parent_row_loss(scores,target).item() < 1e-4


def test_unannotated_parent_child_pair_is_not_globally_labeled_negative():
    scores = torch.tensor([[8.,-8.],[-8.,8.]],requires_grad=True)
    target = torch.tensor([[1.,0.],[0.,0.]])
    assert sparse_parent_row_loss(scores,target).item() < 1e-4


def test_missing_annotations_produce_no_supervision():
    scores = torch.randn(3,4,requires_grad=True)
    loss = sparse_parent_row_loss(scores,torch.zeros_like(scores))
    loss.backward()
    assert loss.item() == 0 and not scores.grad.any()
