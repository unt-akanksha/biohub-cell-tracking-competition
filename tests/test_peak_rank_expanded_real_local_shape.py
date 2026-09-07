from __future__ import annotations

import torch

from research.peak_rank_detection import train_expanded_real_local_shape_detector as variant


def test_local_shape_loss_rewards_a_unique_center_peak() -> None:
    points = torch.tensor([[4.0, 4.0, 4.0]])
    flat = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
    peaked = torch.zeros((1, 1, 8, 8, 8))
    peaked[0, 0, 4, 4, 4] = 4.0
    assert variant.local_peak_shape_loss(peaked, points) < variant.local_peak_shape_loss(
        flat, points
    )


def test_local_shape_loss_protects_neighboring_annotations() -> None:
    logits = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
    points = torch.tensor([[3.0, 3.0, 3.0], [3.0, 3.0, 5.0]])
    loss = variant.local_peak_shape_loss(logits, points)
    loss.backward()
    # The second center is exactly on the first center's shell, but must never
    # receive a negative gradient from that relationship.
    assert logits.grad is not None
    assert float(logits.grad[0, 0, 3, 3, 5]) < 0.0


def test_real_loss_includes_shape_without_dense_real_negatives() -> None:
    logits = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
    prediction = {
        "logits": logits,
        "offsets": torch.zeros((1, 3, 8, 8, 8), requires_grad=True),
        "auxiliary_logits": (
            torch.zeros((1, 1, 2, 2, 2), requires_grad=True),
            torch.zeros((1, 1, 4, 4, 4), requires_grad=True),
        ),
    }
    loss, components = variant.training_loss(
        prediction, torch.tensor([[4.0, 4.0, 4.0]]), complete_labels=False
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert components["local_shape"] > 0.0
    assert components["rank"] == 0.0
    assert variant.RUN_ID.endswith("local-shape-peak-rank-v9")


def test_complete_synthetic_loss_keeps_base_objective() -> None:
    prediction = {
        "logits": torch.zeros((1, 1, 8, 8, 8)),
        "offsets": torch.zeros((1, 3, 8, 8, 8)),
        "auxiliary_logits": (
            torch.zeros((1, 1, 2, 2, 2)),
            torch.zeros((1, 1, 4, 4, 4)),
        ),
    }
    loss, components = variant.training_loss(
        prediction, torch.tensor([[4.0, 4.0, 4.0]]), complete_labels=True
    )
    assert torch.isfinite(loss)
    assert components["local_shape"] == 0.0
