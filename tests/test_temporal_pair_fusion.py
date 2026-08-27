from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from research.temporal_contrastive.appearance_family import (
    COSINE_FAMILY,
    PAIR_FUSION_FAMILY,
    appearance_family,
    build_appearance_model,
    verify_appearance_metadata,
)
from research.temporal_contrastive.pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    PAIR_FEATURE_WIDTH,
    PhysicalPairFusionAssociationModel,
    masked_multi_positive_pair_nll,
    pair_fusion_scores_for_movie,
    pair_logit_metrics,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo


def common_metadata(parameter_count: int) -> dict:
    return {
        "parameter_count": parameter_count,
        "input_channels": 3,
        "temporal_frame_offsets": [-1, 0, 1],
        "checkpoint_weight_source": "optimizer-step exponential moving average",
        "ema_decay": 0.997,
        "division_prior_correction": "class-conditional importance weighting",
        "link_loss_policy": "all-positive supervised contrastive mean-log-probability",
        "real_split_policy": "global deterministic disjoint partition per embryo prefix",
    }


def test_default_pair_fusion_model_matches_frozen_heavy_contract() -> None:
    model = PhysicalPairFusionAssociationModel()

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_PARAMETER_COUNT
    )
    assert model.pair_feature_width == PAIR_FEATURE_WIDTH
    assert model.pair_head[1].in_features == 1_029
    assert model.pair_head[1].out_features == 1_024
    assert model.pair_head[3].out_features == 512
    assert model.pair_head[5].out_features == 128
    assert model.pair_head[7].out_features == 1


def test_appearance_family_contract_supports_legacy_v1_and_exact_v2() -> None:
    legacy = common_metadata(19_221_954)
    assert appearance_family(legacy) == COSINE_FAMILY
    assert verify_appearance_metadata(legacy) == COSINE_FAMILY
    assert type(build_appearance_model(COSINE_FAMILY)).__name__ == (
        "PhysicalPatchAssociationModel"
    )

    pair = {
        **common_metadata(EXPECTED_PARAMETER_COUNT),
        "appearance_family": PAIR_FUSION_FAMILY,
        "run_id": "temporal-patch-pair-fusion-v2",
        "pair_feature_width": 1_029,
        "pair_projection_width": 1_024,
        "pair_hidden_widths": [512, 128],
        "pair_fusion_policy": (
            "candidate-limited source-target-absolute-product-displacement-division MLP"
        ),
        "pair_loss_policy": "all-positive candidate-pair mean-log-probability",
        "embedding_auxiliary_loss_weight": 0.25,
        "pair_chunk_size": 4_096,
    }
    assert verify_appearance_metadata(pair, require_training_run=True) == (
        PAIR_FUSION_FAMILY
    )
    assert isinstance(
        build_appearance_model(PAIR_FUSION_FAMILY),
        PhysicalPairFusionAssociationModel,
    )
    pair["pair_hidden_widths"] = [512, 64]
    with pytest.raises(ValueError, match="pair-fusion architecture"):
        verify_appearance_metadata(pair, require_training_run=True)


def test_candidate_pair_loss_backpropagates_through_encoder_pair_and_division() -> None:
    torch.manual_seed(23)
    model = PhysicalPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    patches = torch.randn(5, 3, 17, 17, 17)
    embeddings, divisions = model(patches)
    candidates = torch.tensor(
        [[True, True, False], [False, True, True]], dtype=torch.bool
    )
    positives = torch.tensor(
        [[True, False, False], [False, False, True]], dtype=torch.bool
    )
    logits = model.candidate_pair_logits(
        embeddings[:2],
        embeddings[2:],
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0]]),
        torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 7.0, 0.0], [0.0, 9.0, 0.0]]
        ),
        divisions[:2],
        candidates,
        candidate_radius_um=16.0,
        chunk_size=2,
    )
    loss = masked_multi_positive_pair_nll(logits, positives, candidates)
    loss.backward()

    assert logits.shape == (2, 3)
    assert torch.equal(torch.isfinite(logits), candidates)
    assert math.isfinite(float(loss.detach()))
    assert model.stem[0].weight.grad is not None
    assert float(model.stem[0].weight.grad.abs().sum()) > 0.0
    assert model.division[-1].weight.grad is not None
    assert float(model.division[-1].weight.grad.abs().sum()) > 0.0
    assert model.pair_head[-1].weight.grad is not None
    assert float(model.pair_head[-1].weight.grad.abs().sum()) > 0.0


def test_pair_logit_metrics_reward_both_daughters_and_arbitrary_mask() -> None:
    logits = torch.tensor(
        [[4.0, 3.0, float("-inf")], [float("-inf"), 1.0, 2.0]]
    )
    candidates = torch.isfinite(logits)
    positives = torch.tensor(
        [[True, True, False], [False, False, True]], dtype=torch.bool
    )

    metrics = pair_logit_metrics(logits, candidates, positives)

    assert metrics["top1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["division_top2"] == 1.0
    assert metrics["division_rows"] == 1


def test_pair_fusion_movie_scores_are_neutral_outside_trackastra_candidates() -> None:
    model = PhysicalPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    with torch.no_grad():
        for parameter in model.pair_head.parameters():
            parameter.zero_()
        model.pair_head[-1].bias.fill_(math.log(4.0))
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
        [[1, 0] + [0] * 14, [0, 1] + [0] * 14, [1, 0] + [0] * 14, [0, 1] + [0] * 14],
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

    scores = pair_fusion_scores_for_movie(
        model,
        video,
        embeddings,
        divisions,
        pair_scores,
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        candidate_radius_um=16.0,
        chunk_size=1,
    )[0]

    np.testing.assert_allclose(scores, [[0.8, 0.5], [0.8, 0.8]], atol=1e-6)


def test_pair_loss_rejects_finite_logits_outside_candidate_mask() -> None:
    logits = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor([[True, False]])
    positives = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="exactly match"):
        masked_multi_positive_pair_nll(logits, positives, candidates)
