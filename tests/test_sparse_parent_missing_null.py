from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research'))
from sparse_parent_missing_null import missing_null_targets
from sparse_parent_loss import sparse_parent_loss


def test_true_null_pushes_all_parents_down_unknown_column_has_no_gradient():
    scores = torch.tensor([[1., 2., 3.], [0., 1., 2.]], requires_grad=True)
    target = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    augmented, truth = missing_null_targets(scores, target, [False, True, False])
    loss = sparse_parent_loss(augmented, truth)
    expected = -(torch.log_softmax(augmented, 0)[0, 0] + torch.log_softmax(augmented, 0)[2, 1])/2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (scores.grad[:, 1] > 0).all()
    assert (scores.grad[:, 2] == 0).all()
    assert scores.grad[0, 0] < 0


def test_no_known_null_is_exact_original_objective():
    scores = torch.randn(2, 3)
    target = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    augmented, truth = missing_null_targets(scores, target, [False]*3)
    torch.testing.assert_close(sparse_parent_loss(augmented, truth),
        sparse_parent_loss(torch.cat([scores, torch.full((1, 3), -4.5)]), torch.cat([target, torch.zeros(1, 3)])))


def test_empty_parent_set_has_real_null_row():
    scores = torch.empty(0, 2, requires_grad=True)
    augmented, truth = missing_null_targets(scores, torch.empty(0, 2), [True, False])
    assert augmented.shape == (1, 2) and truth.shape == (1, 2)
    loss = sparse_parent_loss(augmented, truth)
    assert loss.item() == 0
    loss.backward()


def test_contradictory_null_and_wrong_mask_rejected():
    for mask in ([True, False], [False]):
        with pytest.raises(ValueError):
            missing_null_targets(torch.zeros(2, 2), torch.eye(2), mask)
