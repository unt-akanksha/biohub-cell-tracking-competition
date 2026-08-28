from __future__ import annotations

import math

import pytest
import torch

from research.temporal_contrastive.contextual_pair_fusion import (
    ContextualPairFusionAssociationModel,
    contextual_bidirectional_pair_nll,
)
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    AXIAL_STATISTICS_PER_CHANNEL,
    EXPECTED_PARAMETER_COUNT,
    MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
    AxialProjection2D,
    MultiscaleContextualPairFusionAssociationModel,
    architecture_contract,
    load_contextual_v3_warm_start,
    parameter_count,
)
from research.temporal_contrastive.transition_context import (
    CANDIDATE_CONTEXT_WIDTH,
)


def test_axial_projection_has_deterministic_uniform_initialization() -> None:
    projection = AxialProjection2D(3).eval()
    patches = torch.arange(2 * 3 * 5 * 7 * 9, dtype=torch.float32).reshape(
        2, 3, 5, 7, 9
    )

    result = projection(patches)
    mean = patches.mean(dim=2)

    assert result.shape == (2, 3 * AXIAL_STATISTICS_PER_CHANNEL, 7, 9)
    assert torch.allclose(result[:, -3:], mean)
    assert torch.allclose(result[:, :3], mean)


def test_axial_projection_rejects_wrong_channels() -> None:
    projection = AxialProjection2D(3)
    with pytest.raises(ValueError, match="shape"):
        projection(torch.zeros(1, 2, 5, 7, 9))


def test_default_multiscale_model_is_distinct_and_high_capacity() -> None:
    contract = architecture_contract()

    assert contract["appearance_family"] == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
    assert contract["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert EXPECTED_PARAMETER_COUNT > 2 * 20_747_761
    assert contract["public_code_copied"] is False
    assert contract["public_predictions_copied"] is False
    assert contract["public_leaderboard_used_for_selection"] is False


def test_small_multiscale_model_runs_contextual_loss_and_backward() -> None:
    torch.manual_seed(701)
    model = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
    )
    patches = torch.randn(7, 3, 17, 17, 17)
    candidates = torch.tensor(
        [[True, True, False, False], [False, True, True, True], [False, False, True, True]]
    )
    positives = torch.tensor(
        [[True, False, False, False], [False, False, True, False], [False, False, False, True]]
    )
    source_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 10.0, 0.0]]
    )
    target_coords = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 7.0, 0.0], [0.0, 11.0, 0.0]]
    )
    context = torch.randn(3, 4, CANDIDATE_CONTEXT_WIDTH)
    context[~candidates] = 0
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        embeddings, divisions = model(patches)
        logits = model.candidate_pair_logits(
            embeddings[:3],
            embeddings[3:],
            source_coords,
            target_coords,
            divisions[:3],
            candidates,
            context,
            candidate_radius_um=16.0,
            chunk_size=2,
        )
        loss = contextual_bidirectional_pair_nll(logits, positives, candidates)
        loss = loss + 0.05 * divisions.square().mean()
    loss.float().backward()

    assert embeddings.shape == (7, 16)
    assert divisions.shape == (7,)
    assert torch.isfinite(logits[candidates]).all()
    assert torch.isneginf(logits[~candidates]).all()
    assert math.isfinite(float(loss.detach()))
    assert model.stem[0].weight.grad is not None
    assert float(model.stem[0].weight.grad.abs().sum()) > 0
    assert model.axial_projection.attention_logits[-1].weight.grad is not None
    assert float(
        model.axial_projection.attention_logits[-1].weight.grad.abs().sum()
    ) > 0
    assert model.projection_stem[0].weight.grad is not None
    assert float(model.projection_stem[0].weight.grad.abs().sum()) > 0
    assert model.edge_head[-1].weight.grad is not None
    assert float(model.edge_head[-1].weight.grad.abs().sum()) > 0


def test_v3_warm_start_is_complete_and_numerically_prediction_preserving() -> None:
    torch.manual_seed(1701)
    control = ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    ).eval()
    multiscale = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
    ).eval()
    missing = load_contextual_v3_warm_start(multiscale, control.state_dict())
    patches = torch.randn(4, 3, 17, 17, 17)

    with torch.no_grad():
        control_embeddings, control_divisions = control(patches)
        multiscale_embeddings, multiscale_divisions = multiscale(patches)

    assert missing
    assert all(
        key.startswith(
            (
                "axial_projection.",
                "projection_stem.",
                "projection_encoder.",
                "appearance_adapter.",
                "division_adapter.",
            )
        )
        for key in missing
    )
    assert torch.allclose(
        multiscale_embeddings, control_embeddings, atol=1e-7, rtol=1e-6
    )
    assert torch.equal(multiscale_divisions, control_divisions)


def test_zero_residual_warm_start_unlocks_the_full_axial_path() -> None:
    """The two intentional zero layers must delay, not strand, new features."""

    torch.manual_seed(1907)
    control = ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    model = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
    )
    load_contextual_v3_warm_start(model, control.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    patches = torch.randn(7, 3, 17, 17, 17)
    candidates = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, True],
            [True, False, True, True],
        ]
    )
    positives = torch.tensor(
        [
            [True, False, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )
    source_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 10.0, 0.0]]
    )
    target_coords = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 7.0, 0.0], [0.0, 11.0, 0.0]]
    )
    context = torch.randn(3, 4, CANDIDATE_CONTEXT_WIDTH)
    context[~candidates] = 0

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        embeddings, divisions = model(patches)
        logits = model.candidate_pair_logits(
            embeddings[:3],
            embeddings[3:],
            source_coords,
            target_coords,
            divisions[:3],
            candidates,
            context,
            candidate_radius_um=16.0,
            chunk_size=2,
        )
        loss = contextual_bidirectional_pair_nll(logits, positives, candidates)
        loss = loss + 0.20 * divisions.square().mean()
        loss.backward()

    train_step()
    assert float(model.appearance_adapter[-1].weight.grad.abs().sum()) > 0
    assert float(model.division_adapter[-1].weight.grad.abs().sum()) > 0
    # Both output adapters are exactly zero at the warm start, so upstream
    # axial parameters correctly receive no first-step signal.
    assert float(model.projection_stem[0].weight.grad.abs().sum()) == 0
    optimizer.step()

    train_step()
    assert float(model.projection_stem[0].weight.grad.abs().sum()) > 0
    assert float(
        model.axial_projection.attention_logits[-1].weight.grad.abs().sum()
    ) > 0
    # The attention output layer was also initialized to zero. Its first
    # update now unlocks the depthwise attention feature extractor.
    optimizer.step()

    train_step()
    assert float(
        model.axial_projection.attention_logits[0].weight.grad.abs().sum()
    ) > 0


def test_v3_warm_start_rejects_missing_shared_tensor() -> None:
    control = ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    multiscale = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
    )
    incomplete = dict(control.state_dict())
    del incomplete["stem.0.weight"]
    with pytest.raises(ValueError, match="incomplete"):
        load_contextual_v3_warm_start(multiscale, incomplete)


def test_multiscale_forward_rejects_wrong_channels() -> None:
    model = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
    )
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 2, 9, 9, 9))


def test_parameter_counter_has_no_hidden_default_construction_dependency() -> None:
    model = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
    )
    assert parameter_count(model) == sum(
        parameter.numel() for parameter in model.parameters()
    )
