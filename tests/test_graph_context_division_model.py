from __future__ import annotations

import torch

from research.temporal_contrastive.graph_context_division_model import (
    GraphContextDivisionModel,
    architecture_contract,
    parameter_count,
)


def small_model() -> GraphContextDivisionModel:
    return GraphContextDivisionModel(
        base_channels=8,
        embedding_channels=32,
        projection_base_channels=12,
        context_dimension=64,
        context_heads=4,
        context_layers=2,
        context_feedforward=128,
        dropout=0.0,
        head_widths=(128, 32),
    )


def inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    patches = torch.randn(batch, 3, 3, 17, 17, 17)
    geometry = torch.randn(batch, 9)
    context = torch.randn(batch, 43, 8)
    context[:, :, 5:] = 0.0
    context[:, 0, 5] = 1.0
    context[:, 1:3, 6] = 1.0
    context[:, 3:17, 7] = 1.0
    mask = torch.zeros(batch, 43, dtype=torch.bool)
    mask[:, :17] = True
    return patches, geometry, context, mask


def test_default_architecture_is_large_and_has_no_positional_shortcut() -> None:
    contract = architecture_contract()

    assert contract["parameter_count"] > 70_000_000
    assert contract["backbone_parameter_count"] == 46_386_607
    assert contract["graph_context_parameter_count"] > 25_000_000
    assert contract["context_layers"] == 8
    assert contract["context_dimension"] == 512
    assert contract["learned_positional_embedding"] is False
    assert contract["permutation_invariant_context"] is True
    assert contract["daughter_order_invariant"] is True
    assert contract["public_code_copied"] is False
    assert contract["public_leaderboard_used_for_selection"] is False


def test_forward_is_daughter_order_invariant_and_backpropagates() -> None:
    torch.manual_seed(37)
    model = small_model().eval()
    patches, geometry, context, mask = inputs()
    first = model(patches, geometry, context, mask)
    swapped_patches = patches[:, [0, 2, 1]]
    second = model(swapped_patches, geometry, context, mask)

    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    first.sum().backward()
    assert model.backbone.stem[0].weight.grad is not None
    assert model.context_encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.ranking_head[-1].weight.grad is not None
    assert parameter_count(model) > parameter_count(model.backbone)


def test_context_token_permutation_does_not_change_score() -> None:
    torch.manual_seed(71)
    model = small_model().eval()
    patches, geometry, context, mask = inputs(batch=1)
    baseline = model(patches, geometry, context, mask)
    permutation = torch.randperm(43)
    permuted = model(
        patches,
        geometry,
        context[:, permutation],
        mask[:, permutation],
    )

    torch.testing.assert_close(baseline, permuted, rtol=1e-5, atol=1e-6)


def test_invalid_context_mask_is_rejected() -> None:
    model = small_model()
    patches, geometry, context, mask = inputs(batch=1)
    mask[:, 1] = False

    try:
        model(patches, geometry, context, mask)
    except ValueError as error:
        assert "anchor tokens" in str(error)
    else:
        raise AssertionError("invalid graph-context anchors were accepted")
