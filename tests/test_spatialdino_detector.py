from __future__ import annotations

import pytest
import torch

from research.spatialdino_association.encoder import SpatialDinoViTS8
from research.spatialdino_detection.model import (
    HybridSpatialDinoDetector,
    set_detector_training_phase,
)


class TinySpatialDino(SpatialDinoViTS8):
    embed_dim = 24
    depth = 12
    patch_size = 8

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.patch_embed = torch.nn.Conv3d(1, self.embed_dim, kernel_size=8, stride=8)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=1),
                torch.nn.GELU(),
            )
            for _ in range(self.depth)
        )
        self.norm = torch.nn.GroupNorm(1, self.embed_dim)

    def forward_intermediates(self, volume, *, block_indices=(2, 5, 8, 11)):
        values = self.patch_embed(volume)
        output = []
        selected = set(block_indices)
        for index, block in enumerate(self.blocks):
            values = values + block(values)
            if index in selected:
                output.append(self.norm(values) if index == self.depth - 1 else values)
        return tuple(output)


def test_hybrid_detector_preserves_full_spatial_resolution() -> None:
    model = HybridSpatialDinoDetector(TinySpatialDino(), widths=(8, 16, 24, 32))
    torch.testing.assert_close(model.heatmap_head.bias, torch.full_like(model.heatmap_head.bias, -4.0))
    assert float(model.heatmap_head.weight.detach().std()) < 0.002
    volume = torch.randn(1, 1, 16, 16, 16)
    logits = model(volume)
    assert logits.shape == (1, 1, 16, 16, 16)
    assert torch.isfinite(logits).all()


def test_detector_rejects_non_aligned_or_multichannel_input() -> None:
    model = HybridSpatialDinoDetector(TinySpatialDino(), widths=(8, 16, 24, 32))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 2, 16, 16, 16))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros(1, 1, 16, 16, 17))


def test_training_phase_freezes_encoder_then_unfreezes_only_tail() -> None:
    model = HybridSpatialDinoDetector(TinySpatialDino(), widths=(8, 16, 24, 32))
    frozen = set_detector_training_phase(model, unfreeze_last_encoder_blocks=0)
    assert frozen["encoder_trainable_parameters"] == 0
    assert frozen["decoder_trainable_parameters"] > 0
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())

    tail = set_detector_training_phase(model, unfreeze_last_encoder_blocks=2)
    assert tail["encoder_trainable_parameters"] > 0
    assert all(
        parameter.requires_grad
        for block in model.encoder.blocks[-2:]
        for parameter in block.parameters()
    )
    assert all(
        not parameter.requires_grad
        for block in model.encoder.blocks[:-2]
        for parameter in block.parameters()
    )


def test_hybrid_detector_completes_finite_optimizer_step() -> None:
    torch.manual_seed(19)
    model = HybridSpatialDinoDetector(TinySpatialDino(), widths=(8, 16, 24, 32))
    set_detector_training_phase(model, unfreeze_last_encoder_blocks=0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    before = model.heatmap_head.weight.detach().clone()
    logits = model(torch.randn(1, 1, 16, 16, 16))
    target = torch.zeros_like(logits)
    target[:, :, 8, 8, 8] = 1.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    assert not torch.equal(before, model.heatmap_head.weight.detach())
