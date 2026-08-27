from __future__ import annotations

import numpy as np
import pytest
import torch

from research.temporal_contrastive.model import (
    TemporalFusionHead,
    masked_link_info_nce,
    masked_multi_positive_info_nce,
)
from research.temporal_contrastive.patch_model import (
    PhysicalPatchAssociationModel,
    physical_candidate_masks,
    sample_physical_patches,
)
from research.temporal_contrastive.train_dual_fold_patch import (
    aggregate_metrics,
    division_prior_corrected_bce,
    prepare_transition,
    synthetic_split,
    transition_metrics,
    update_ema_model,
)
from research.temporal_contrastive.appearance_blend import (
    appearance_scores_for_movie,
    blend_pair_scores,
    division_logits_for_movie,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo
from research.temporal_contrastive.calibrate_dual_fold_blend import (
    APPEARANCE_WEIGHTS,
    DIVISION_WEIGHTS,
    association_metrics_for_video,
    ranking_metrics_for_video,
    select_weight,
)


def test_temporal_head_outputs_normalized_embedding_and_sparse_division_prior() -> None:
    torch.manual_seed(7)
    head = TemporalFusionHead(8, hidden_channels=16, embedding_channels=6)
    current = torch.randn(2, 8, 4, 4, 4)
    following = torch.randn_like(current)

    embeddings, division_logits = head(current, following)

    assert embeddings.shape == (2, 6, 4, 4, 4)
    assert division_logits.shape == (2, 1, 4, 4, 4)
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(2, 4, 4, 4),
        atol=1e-5,
    )
    assert float(division_logits.detach().mean()) < -3.0


def test_masked_info_nce_rewards_correct_link_similarity() -> None:
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    positives = torch.tensor([0, 1])
    candidates = torch.ones((2, 3), dtype=torch.bool)
    aligned = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reversed_sources = aligned.flip(0)

    good = masked_link_info_nce(aligned, targets, positives, candidates)
    bad = masked_link_info_nce(reversed_sources, targets, positives, candidates)

    assert float(good) < float(bad)


def test_masked_info_nce_excludes_unknown_sources_but_keeps_gradients() -> None:
    sources = torch.randn(3, 4, requires_grad=True)
    targets = torch.randn(3, 4)
    positives = torch.tensor([0, -1, 2])
    candidates = torch.ones((3, 3), dtype=torch.bool)

    loss = masked_link_info_nce(sources, targets, positives, candidates)
    loss.backward()

    assert torch.isfinite(loss)
    assert sources.grad is not None
    assert torch.count_nonzero(sources.grad[1]) == 0


def test_masked_info_nce_rejects_missing_positive_candidate() -> None:
    sources = torch.randn(1, 4)
    targets = torch.randn(2, 4)
    candidates = torch.tensor([[False, True]])

    with pytest.raises(ValueError, match="ground-truth link"):
        masked_link_info_nce(sources, targets, torch.tensor([0]), candidates)


def test_multi_positive_info_nce_does_not_treat_a_second_daughter_as_negative() -> None:
    sources = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    positives = torch.tensor(
        [[True, True, False, False], [False, False, True, False]]
    )
    candidates = torch.ones_like(positives)

    good = masked_multi_positive_info_nce(sources, targets, positives, candidates)
    bad = masked_multi_positive_info_nce(sources.flip(0), targets, positives, candidates)

    assert float(good) < float(bad)


def test_physical_patch_sampling_matches_anisotropic_and_isotropic_views() -> None:
    z, y, x = torch.meshgrid(
        torch.arange(9), torch.arange(17), torch.arange(17), indexing="ij"
    )
    anisotropic = z.float() * 2.0 + y.float() + x.float()
    patch = sample_physical_patches(
        anisotropic,
        [[4.0, 8.0, 8.0]],
        voxel_size_zyx_um=(2.0, 1.0, 1.0),
        output_shape_zyx=(5, 5, 5),
        half_extent_zyx_um=(4.0, 2.0, 2.0),
    )

    assert patch.shape == (1, 1, 5, 5, 5)
    assert torch.isfinite(patch).all()
    assert abs(float(patch.mean())) < 1e-5
    assert 0.9 < float(patch.std()) < 1.1


def test_candidate_mask_fails_closed_when_radius_drops_a_true_jump() -> None:
    source = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.asarray([[0.0, 2.0, 0.0], [0.0, 20.0, 0.0]], dtype=np.float32)
    candidates, positives = physical_candidate_masks(
        source,
        target,
        np.asarray([[0, 0]]),
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        radius_um=3.0,
    )
    assert candidates.tolist() == [[True, False]]
    assert positives.tolist() == [[True, False]]

    with pytest.raises(ValueError, match="omitted"):
        physical_candidate_masks(
            source,
            target,
            np.asarray([[0, 1]]),
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
            radius_um=3.0,
        )


def test_patch_association_model_has_normalized_embeddings_and_sparse_divisions() -> None:
    torch.manual_seed(11)
    model = PhysicalPatchAssociationModel(base_channels=8, embedding_channels=16)
    patches = torch.randn(3, 1, 17, 17, 17)

    embeddings, divisions = model(patches)
    logits = model.pair_logits(embeddings[:2], embeddings[1:])

    assert embeddings.shape == (3, 16)
    assert divisions.shape == (3,)
    assert logits.shape == (2, 2)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3), atol=1e-5)
    assert float(divisions.detach().mean()) < -3.0


def test_default_patch_model_is_the_declared_heavy_candidate() -> None:
    model = PhysicalPatchAssociationModel()

    assert sum(parameter.numel() for parameter in model.parameters()) == 19_218_498
    assert model.projection[-1].out_features == 256


def test_full_model_ema_averages_optimizer_weights() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    ema = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
        ema.weight.zero_()

    update_ema_model(ema, model, decay=0.75)

    torch.testing.assert_close(ema.weight, torch.full_like(ema.weight, 0.5))
    with pytest.raises(ValueError, match="EMA decay"):
        update_ema_model(ema, model, decay=1.0)


def test_synthetic_division_prior_correction_preserves_negative_learning() -> None:
    logits = torch.zeros(2)
    positive = division_prior_corrected_bce(
        logits, torch.ones(2), synthetic=True
    )
    negative = division_prior_corrected_bce(
        logits, torch.zeros(2), synthetic=True
    )
    real = division_prior_corrected_bce(logits, torch.zeros(2), synthetic=False)

    assert float(positive) < 0.05
    assert float(negative) > 0.70
    assert float(negative) > float(real)


def test_transition_sampler_preserves_division_positives_and_hard_negatives() -> None:
    times = np.asarray([0, 0, 1, 1, 1], dtype=np.int32)
    coords = np.asarray(
        [[0, 0, 0], [0, 8, 0], [0, 1, 0], [0, 2, 0], [0, 7, 0]],
        dtype=np.float32,
    )
    edges = np.asarray([[0, 2], [0, 3], [1, 4]], dtype=np.int64)
    batch = prepare_transition(
        times,
        coords,
        edges,
        timepoint=0,
        voxel_size_zyx_um=(1, 1, 1),
        radius_um=10,
        max_sources=2,
        max_targets=3,
        rng=np.random.default_rng(3),
    )

    assert batch.positive_mask.sum(axis=1).tolist() == [2, 1]
    assert batch.division_target.tolist() == [1.0, 0.0]
    assert np.all(batch.candidate_mask.sum(axis=1) > batch.positive_mask.sum(axis=1))


def test_transition_metrics_reward_correct_division_ranking() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    positives = torch.tensor(
        [[True, True, False], [False, False, True]], dtype=torch.bool
    )
    candidates = torch.ones_like(positives)

    row = transition_metrics(source, target, candidates, positives)
    pooled = aggregate_metrics([row, row])

    assert row["top1"] == 1.0
    assert row["division_top2"] == 1.0
    assert pooled["top1"] == 1.0
    assert pooled["division_rows"] == 2


def test_zero_weight_blend_is_exact_and_positive_weight_changes_ambiguity() -> None:
    track = np.asarray([[0.55, 0.54], [0.70, -np.inf]], dtype=np.float32)
    appearance = np.asarray([[0.1, 0.9], [0.5, -0.2]], dtype=np.float32)

    unchanged = blend_pair_scores(track, appearance, appearance_weight=0.0)
    blended = blend_pair_scores(track, appearance, appearance_weight=0.25)

    np.testing.assert_array_equal(unchanged, track)
    assert blended[0, 1] > blended[0, 0]
    assert np.isneginf(blended[1, 1])


def test_division_evidence_can_restore_a_second_daughter() -> None:
    video = GraphVideo(
        "fixture",
        node_ids=np.asarray([1, 2, 3, 4]),
        times=np.asarray([0, 1, 1, 1]),
        coords_voxel=np.zeros((4, 3), dtype=np.float32),
        edges=np.asarray([[1, 2], [1, 3]], dtype=np.int64),
    )
    track = np.asarray([[0.90, 0.44, 0.10]], dtype=np.float32)
    appearance = np.zeros_like(track)
    control = {0: (np.asarray([1]), np.asarray([2, 3, 4]), track)}
    boosted = {
        0: (
            np.asarray([1]),
            np.asarray([2, 3, 4]),
            blend_pair_scores(
                track,
                appearance,
                appearance_weight=0.05,
                source_division_logits=np.asarray([2.0], dtype=np.float32),
                division_weight=0.20,
            ),
        )
    }

    control_metrics = association_metrics_for_video(video, control)
    boosted_metrics = association_metrics_for_video(video, boosted)
    assert control_metrics["division_jaccard"] == 0.0
    assert control_metrics["composite"] == 0.5
    assert boosted_metrics["division_jaccard"] == 1.0
    assert boosted_metrics["composite"] == 1.1


def test_division_logits_align_arbitrary_source_identifiers() -> None:
    video = GraphVideo(
        "fixture",
        node_ids=np.asarray([20, 10, 40, 30]),
        times=np.asarray([0, 0, 1, 1]),
        coords_voxel=np.zeros((4, 3), dtype=np.float32),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    pair_scores = {
        0: (
            np.asarray([10, 20]),
            np.asarray([30, 40]),
            np.full((2, 2), 0.5, dtype=np.float32),
        )
    }

    aligned = division_logits_for_movie(
        video, np.asarray([2.0, -3.0, 0.0, 0.0]), pair_scores
    )

    np.testing.assert_array_equal(aligned[0], [-3.0, 2.0])


def test_appearance_scores_align_arbitrary_node_identifiers() -> None:
    video = GraphVideo(
        "fixture",
        node_ids=np.asarray([20, 10, 40, 30]),
        times=np.asarray([0, 0, 1, 1]),
        coords_voxel=np.zeros((4, 3), dtype=np.float32),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    embeddings = np.asarray(
        [[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32
    )
    pair_scores = {
        0: (
            np.asarray([10, 20]),
            np.asarray([30, 40]),
            np.full((2, 2), 0.5, dtype=np.float32),
        )
    }

    appearance = appearance_scores_for_movie(video, embeddings, pair_scores)[0]

    np.testing.assert_allclose(appearance, [[1, 0], [0, 1]])


def test_synthetic_split_reads_public_manifest_schema_deterministically(tmp_path) -> None:
    import json

    sequence_dir = tmp_path / "sequences"
    sequence_dir.mkdir()
    records = []
    for index in range(4):
        path = sequence_dir / f"seq_{index:04d}.npz"
        path.write_bytes(b"fixture")
        records.append(
            {"file": f"sequences/{path.name}", "T": 6, "n_edges": index + 1}
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sequences": records}), encoding="utf-8")

    observed_manifest, training, validation = synthetic_split(
        tmp_path, validation_count=1, training_count=2
    )

    assert observed_manifest == manifest
    assert len(training) == 2
    assert len(validation) == 1
    assert set(training).isdisjoint(validation)


def test_blend_calibration_ranking_uses_true_division_children() -> None:
    video = GraphVideo(
        "fixture",
        node_ids=np.asarray([1, 2, 3, 4]),
        times=np.asarray([0, 1, 1, 1]),
        coords_voxel=np.zeros((4, 3), dtype=np.float32),
        edges=np.asarray([[1, 2], [1, 3]], dtype=np.int64),
    )
    pair_scores = {
        0: (
            np.asarray([1]),
            np.asarray([2, 3, 4]),
            np.asarray([[0.9, 0.8, 0.1]], dtype=np.float32),
        )
    }

    metrics = ranking_metrics_for_video(video, pair_scores)

    assert metrics["top1"] == 1.0
    assert metrics["division_top2"] == 1.0
    assert metrics["division_rows"] == 1

    omitted = {
        0: (
            np.asarray([1]),
            np.asarray([2, 3, 4]),
            np.asarray([[0.9, -np.inf, 0.1]], dtype=np.float32),
        )
    }
    with pytest.raises(RuntimeError, match="omitted"):
        ranking_metrics_for_video(video, omitted)


def test_blend_selection_requires_gain_and_movie_floor() -> None:
    rows = []
    for appearance_weight in APPEARANCE_WEIGHTS:
        for division_weight in DIVISION_WEIGHTS:
            gain = 0.0 if appearance_weight == 0 else 0.002
            movie_a_gain = -0.001 if appearance_weight == 0.10 else gain
            rows.append(
                {
                    "appearance_weight": appearance_weight,
                    "division_weight": division_weight,
                    "pooled": {"composite": 0.80 + gain},
                    "by_movie": [
                        {"stem": "a", "composite": 0.80 + movie_a_gain},
                        {"stem": "b", "composite": 0.80 + gain},
                    ],
                }
            )

    selected = select_weight(rows)

    assert selected["improved"] is True
    assert selected["selected_weight"] == 0.05
    assert selected["selected_division_weight"] == 0.0
