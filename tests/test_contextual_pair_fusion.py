from __future__ import annotations

import torch
import pytest

from research.temporal_contrastive.appearance_family import (
    candidate_appearance_family,
    build_appearance_model,
    verify_appearance_metadata,
)
from research.temporal_contrastive.contextual_pair_fusion import (
    CONTEXTUAL_PAIR_FEATURE_WIDTH,
    EDGE_SET_FEATURE_WIDTH,
    EXPECTED_PARAMETER_COUNT,
    CONTEXTUAL_PAIR_FUSION_FAMILY,
    CONTEXTUAL_PAIR_LOSS_POLICY,
    CONTEXTUAL_PAIR_POLICY,
    EDGE_HEAD_HIDDEN_WIDTHS,
    EDGE_TOKEN_WIDTH,
    RECIPROCAL_PARENT_LOSS_WEIGHT,
    TRANSITION_CONTEXT_POLICY,
    ContextualPairFusionAssociationModel,
    contextual_pair_fusion_scores_for_movie,
)
from research.temporal_contrastive.transition_context import (
    CANDIDATE_CONTEXT_WIDTH,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo
import numpy as np


def inputs():
    torch.manual_seed(19)
    source_embeddings = torch.randn(3, 256, requires_grad=True)
    target_embeddings = torch.randn(4, 256, requires_grad=True)
    source_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, 3.0, 2.0], [8.0, 5.0, 4.0]]
    )
    target_coords = torch.tensor(
        [[1.0, 0.0, 0.0], [5.0, 3.0, 2.0], [9.0, 5.0, 4.0], [7.0, 7.0, 7.0]]
    )
    divisions = torch.tensor([0.2, -0.7, 1.3])
    candidates = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, True],
            [False, False, True, True],
        ]
    )
    context = torch.randn(3, 4, CANDIDATE_CONTEXT_WIDTH)
    context[~candidates] = 0.0
    return (
        source_embeddings,
        target_embeddings,
        source_coords,
        target_coords,
        divisions,
        candidates,
        context,
    )


def test_contextual_head_has_frozen_widths_and_finite_backward() -> None:
    model = ContextualPairFusionAssociationModel()
    values = inputs()
    logits = model.candidate_pair_logits(*values, chunk_size=3)

    assert model.pair_feature_width == CONTEXTUAL_PAIR_FEATURE_WIDTH == 1_047
    assert EDGE_SET_FEATURE_WIDTH == 1_536
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT
    assert torch.isfinite(logits[values[5]]).all()
    assert torch.isneginf(logits[~values[5]]).all()
    loss = logits[values[5]].square().mean()
    loss.backward()
    populated = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert populated
    assert all(torch.isfinite(gradient).all() for gradient in populated)


def test_contextual_head_scatter_is_autocast_dtype_safe() -> None:
    model = ContextualPairFusionAssociationModel()
    values = inputs()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model.candidate_pair_logits(*values, chunk_size=3)

    assert logits.dtype == values[0].dtype == torch.float32
    assert torch.isfinite(logits[values[5]]).all()
    assert torch.isneginf(logits[~values[5]]).all()
    loss = logits[values[5]].square().mean()
    loss.backward()
    populated = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert populated
    assert all(torch.isfinite(gradient).all() for gradient in populated)


def test_contextual_family_contract_is_exact_and_buildable() -> None:
    payload = {
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "run_id": "temporal-contextual-pair-fusion-v3",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "input_channels": 3,
        "temporal_frame_offsets": [-1, 0, 1],
        "checkpoint_weight_source": "optimizer-step exponential moving average",
        "ema_decay": 0.997,
        "division_prior_correction": "class-conditional importance weighting",
        "link_loss_policy": "all-positive supervised contrastive mean-log-probability",
        "real_split_policy": "global deterministic disjoint partition per embryo prefix",
        "candidate_context_width": CANDIDATE_CONTEXT_WIDTH,
        "contextual_pair_feature_width": CONTEXTUAL_PAIR_FEATURE_WIDTH,
        "edge_token_width": EDGE_TOKEN_WIDTH,
        "edge_set_feature_width": EDGE_SET_FEATURE_WIDTH,
        "edge_head_hidden_widths": list(EDGE_HEAD_HIDDEN_WIDTHS),
        "contextual_pair_policy": CONTEXTUAL_PAIR_POLICY,
        "transition_context_policy": TRANSITION_CONTEXT_POLICY,
        "pair_loss_policy": CONTEXTUAL_PAIR_LOSS_POLICY,
        "reciprocal_parent_loss_weight": RECIPROCAL_PARENT_LOSS_WEIGHT,
        "embedding_auxiliary_loss_weight": 0.25,
        "pair_chunk_size": 4_096,
    }

    assert verify_appearance_metadata(payload, require_training_run=True) == (
        CONTEXTUAL_PAIR_FUSION_FAMILY
    )
    assert isinstance(
        build_appearance_model(CONTEXTUAL_PAIR_FUSION_FAMILY),
        ContextualPairFusionAssociationModel,
    )
    assert candidate_appearance_family(
        "trackastra_contextual_pair_fusion_blend"
    ) == CONTEXTUAL_PAIR_FUSION_FAMILY
    payload["transition_context_policy"] = "mutated"
    with pytest.raises(ValueError, match="contextual pair-fusion architecture"):
        verify_appearance_metadata(payload, require_training_run=True)


def test_contextual_logits_are_source_and_target_permutation_equivariant() -> None:
    model = ContextualPairFusionAssociationModel().eval()
    values = inputs()
    source_order = torch.tensor([2, 0, 1])
    target_order = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        original = model.candidate_pair_logits(*values)
        permuted = model.candidate_pair_logits(
            values[0][source_order],
            values[1][target_order],
            values[2][source_order],
            values[3][target_order],
            values[4][source_order],
            values[5][source_order][:, target_order],
            values[6][source_order][:, target_order],
        )
    expected = original[source_order][:, target_order]
    assert torch.allclose(permuted[torch.isfinite(permuted)], expected[torch.isfinite(expected)])
    assert torch.equal(torch.isfinite(permuted), torch.isfinite(expected))


def test_contextual_head_rejects_wrong_context_width() -> None:
    model = ContextualPairFusionAssociationModel()
    values = list(inputs())
    values[-1] = values[-1][..., :-1]
    try:
        model.candidate_pair_logits(*values)
    except ValueError as error:
        assert "wrong dense shape" in str(error)
    else:
        raise AssertionError("wrong candidate context width was accepted")


def test_contextual_movie_scores_use_images_and_remain_neutral_off_candidates() -> None:
    model = ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    with torch.no_grad():
        for parameter in model.edge_token.parameters():
            parameter.zero_()
        for parameter in model.edge_head.parameters():
            parameter.zero_()
        model.edge_head[-1].bias.fill_(torch.log(torch.tensor(4.0)))
    video = GraphVideo(
        "fixture",
        node_ids=np.asarray([20, 10, 40, 30]),
        times=np.asarray([0, 0, 1, 1]),
        coords_voxel=np.asarray(
            [[0, 8, 0], [0, 0, 0], [0, 7, 0], [0, 1, 0]],
            dtype=np.float32,
        ),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    embeddings = np.asarray(
        [
            [1, 0] + [0] * 14,
            [0, 1] + [0] * 14,
            [1, 0] + [0] * 14,
            [0, 1] + [0] * 14,
        ],
        dtype=np.float32,
    )
    divisions = np.asarray([2.0, -3.0, 0.0, 0.0], dtype=np.float32)
    pair_scores = {
        0: (
            np.asarray([10, 20]),
            np.asarray([30, 40]),
            np.asarray([[0.5, -np.inf], [0.4, 0.6]], dtype=np.float32),
        )
    }
    rng = np.random.default_rng(17)
    first = rng.normal(size=(17, 21, 25)).astype(np.float32)
    image = np.stack((first, np.roll(first, (1, -2, 3), axis=(0, 1, 2))))

    scores = contextual_pair_fusion_scores_for_movie(
        model,
        video,
        image,
        embeddings,
        divisions,
        pair_scores,
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        candidate_radius_um=16.0,
        chunk_size=1,
    )[0]

    np.testing.assert_allclose(scores, [[0.8, 0.5], [0.8, 0.8]], atol=1e-6)
