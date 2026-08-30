from __future__ import annotations

import pytest
import torch

from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.relational_division_model import (
    RelationalDivisionModel,
    architecture_contract,
    load_backbone_checkpoint,
    normalize_geometry,
)


def small_model() -> RelationalDivisionModel:
    return RelationalDivisionModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
        dropout=0.0,
    )


def test_default_relational_model_is_heavier_than_shared_backbone() -> None:
    contract = architecture_contract()

    assert contract["backbone_parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert contract["parameter_count"] > 48_000_000
    assert contract["relational_parameter_count"] > 1_500_000
    assert contract["daughter_order_invariant"] is True
    assert contract["public_code_copied"] is False
    assert contract["public_predictions_copied"] is False


def test_forward_is_daughter_order_invariant_and_backpropagates() -> None:
    torch.manual_seed(311)
    model = small_model().eval()
    patches = torch.randn(3, 3, 3, 17, 17, 17)
    geometry = torch.tensor(
        [
            [5.0, 8.0, 4.0, 1.0, -0.8, 0.8, 4.0, 2.0, 1.0],
            [4.0, 7.0, 6.0, 2.0, -0.4, 0.7, 3.5, float("nan"), float("nan")],
            [8.0, 9.0, 3.0, 1.5, -0.9, 0.4, 3.2, 1.0, 2.0],
        ]
    )
    swapped_patches = patches[:, [0, 2, 1]]
    swapped_geometry = geometry.clone()
    swapped_geometry[:, 0], swapped_geometry[:, 2] = (
        geometry[:, 2],
        geometry[:, 0],
    )

    original = model(patches, geometry)
    swapped = model(swapped_patches, swapped_geometry)
    loss = original.square().mean()
    loss.backward()

    assert original.shape == (3,)
    assert torch.allclose(original, swapped, atol=1e-6, rtol=1e-6)
    assert model.backbone.stem[0].weight.grad is not None
    assert float(model.backbone.stem[0].weight.grad.abs().sum()) > 0
    assert model.relational_head[-1].weight.grad is not None
    assert float(model.relational_head[-1].weight.grad.abs().sum()) > 0


def test_geometry_normalization_is_symmetric_and_marks_missing_values() -> None:
    geometry = torch.tensor(
        [[5.0, 8.0, 3.0, 1.0, -0.5, 0.8, 4.0, float("nan"), 2.0]]
    )
    swapped = geometry.clone()
    swapped[:, 0], swapped[:, 2] = geometry[:, 2], geometry[:, 0]

    first = normalize_geometry(geometry)
    second = normalize_geometry(swapped)

    assert first.shape == (1, 18)
    assert torch.equal(first, second)
    assert first[0, 9 + 7] == 1.0


def test_strict_backbone_warm_start_rejects_missing_tensor() -> None:
    model = small_model()
    backbone = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16, projection_base_channels=8
    )
    state = dict(backbone.state_dict())
    load_backbone_checkpoint(model, state)
    del state["stem.0.weight"]

    with pytest.raises(ValueError, match="keys changed"):
        load_backbone_checkpoint(model, state)


def test_forward_rejects_wrong_relational_layout() -> None:
    model = small_model()

    with pytest.raises(ValueError, match="relational patches"):
        model(torch.zeros(2, 2, 3, 17, 17, 17), torch.zeros(2, 9))
