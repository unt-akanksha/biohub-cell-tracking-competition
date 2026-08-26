from __future__ import annotations

import numpy as np
import pytest
import torch

from research.spatialdino_detection.distillation import (
    build_selective_distillation_targets,
    selective_distillation_bce,
)


def test_selective_targets_discount_teacher_disagreement() -> None:
    primary = np.zeros((2, 2, 2), dtype=np.float32)
    secondary = np.zeros_like(primary)
    primary[0, 0, 0] = secondary[0, 0, 0] = 0.8
    primary[1, 1, 1] = 0.8
    secondary[1, 1, 1] = 0.2
    targets = build_selective_distillation_targets(primary, secondary)
    assert targets.support_mask.sum() == 2
    assert targets.weights[0, 0, 0] > targets.weights[1, 1, 1]
    assert targets.weights[0, 1, 0] == 0.0
    assert targets.probability[1, 1, 1] == pytest.approx(0.5)


def test_distillation_loss_prefers_matching_soft_logits() -> None:
    probability = torch.tensor([[[[[0.2, 0.8]]]]])
    weights = torch.ones_like(probability)
    matching = torch.logit(probability)
    inverted = torch.logit(1.0 - probability)
    assert selective_distillation_bce(matching, probability, weights) < selective_distillation_bce(
        inverted, probability, weights
    )


def test_empty_support_has_differentiable_zero_loss() -> None:
    logits = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
    probability = torch.zeros_like(logits)
    weights = torch.zeros_like(logits)
    loss = selective_distillation_bce(logits, probability, weights)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert logits.grad is not None


def test_invalid_teacher_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="same 3D"):
        build_selective_distillation_targets(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_selective_distillation_targets(
            np.full((2, 2, 2), 1.1), np.zeros((2, 2, 2))
        )
