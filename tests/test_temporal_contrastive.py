from __future__ import annotations

from pathlib import Path

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
    temporal_context_volume,
)
from research.temporal_contrastive.train_dual_fold_patch import (
    aggregate_metrics,
    division_prior_corrected_bce,
    finish_optimizer_step,
    prepare_transition,
    real_prefix_partition,
    synthetic_split,
    transition_metrics,
    update_ema_model,
)
from research.temporal_contrastive.appearance_blend import (
    appearance_scores_for_movie,
    blend_pair_scores,
    division_logits_for_movie,
    extract_movie_embeddings,
    extract_reciprocal_movie_embeddings,
    reciprocal_movie_evidence,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo
from research.temporal_contrastive.calibrate_dual_fold_blend import (
    APPEARANCE_WEIGHTS,
    DIVISION_WEIGHTS,
    ENSEMBLE_MODES,
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


def test_multi_positive_info_nce_requires_both_daughters_to_score_well() -> None:
    source = torch.tensor([[1.0, 0.0]])
    candidates = torch.ones((1, 3), dtype=torch.bool)
    positives = torch.tensor([[True, True, False]])
    both_daughters = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    one_daughter_only = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])

    complete = masked_multi_positive_info_nce(
        source, both_daughters, positives, candidates
    )
    incomplete = masked_multi_positive_info_nce(
        source, one_daughter_only, positives, candidates
    )

    assert float(complete) < float(incomplete)


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


def test_temporal_context_clamps_boundaries_and_shares_physical_grid() -> None:
    movie = np.stack(
        [np.full((5, 7, 7), value, dtype=np.float32) for value in (1, 2, 4)]
    )

    first = temporal_context_volume(movie, 0)
    last = temporal_context_volume(movie, 2)
    patches = sample_physical_patches(
        first,
        [[2.0, 3.0, 3.0]],
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        output_shape_zyx=(3, 3, 3),
        half_extent_zyx_um=(1.0, 1.0, 1.0),
    )

    assert first[:, 0, 0, 0].tolist() == [1.0, 1.0, 2.0]
    assert last[:, 0, 0, 0].tolist() == [2.0, 4.0, 4.0]
    assert patches.shape == (1, 3, 3, 3, 3)
    assert torch.isfinite(patches).all()


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
    patches = torch.randn(3, 3, 17, 17, 17)

    embeddings, divisions = model(patches)
    logits = model.pair_logits(embeddings[:2], embeddings[1:])

    assert embeddings.shape == (3, 16)
    assert divisions.shape == (3,)
    assert logits.shape == (2, 2)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3), atol=1e-5)
    assert float(divisions.detach().mean()) < -3.0


def test_default_patch_model_is_the_declared_heavy_candidate() -> None:
    model = PhysicalPatchAssociationModel()

    assert sum(parameter.numel() for parameter in model.parameters()) == 19_221_954
    assert model.input_channels == 3
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


def test_partial_gradient_accumulation_is_rescaled_before_update() -> None:
    full = torch.nn.Linear(1, 1, bias=False)
    partial = torch.nn.Linear(1, 1, bias=False)
    full.weight.data.zero_()
    partial.weight.data.zero_()
    full_ema = torch.nn.Linear(1, 1, bias=False)
    partial_ema = torch.nn.Linear(1, 1, bias=False)
    full_ema.weight.data.zero_()
    partial_ema.weight.data.zero_()
    full_optimizer = torch.optim.SGD(full.parameters(), lr=0.1)
    partial_optimizer = torch.optim.SGD(partial.parameters(), lr=0.1)
    full_scaler = torch.amp.GradScaler("cpu")
    partial_scaler = torch.amp.GradScaler("cpu")

    for _ in range(2):
        full_scaler.scale((full(torch.ones(1, 1)).sum() - 1.0) ** 2 / 2).backward()
    partial_scaler.scale(
        (partial(torch.ones(1, 1)).sum() - 1.0) ** 2 / 2
    ).backward()
    finish_optimizer_step(
        full,
        full_ema,
        full_optimizer,
        full_scaler,
        accumulated_batches=2,
        gradient_accumulation=2,
        ema_decay=0.0,
    )
    finish_optimizer_step(
        partial,
        partial_ema,
        partial_optimizer,
        partial_scaler,
        accumulated_batches=1,
        gradient_accumulation=2,
        ema_decay=0.0,
    )

    torch.testing.assert_close(partial.weight, full.weight)
    torch.testing.assert_close(partial_ema.weight, full_ema.weight)


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


def test_reciprocal_appearance_mean_combines_both_clean_models() -> None:
    primary_scores = {0: np.asarray([[1.0, 0.0]], dtype=np.float32)}
    peer_scores = {0: np.asarray([[0.0, 1.0]], dtype=np.float32)}
    primary_divisions = {0: np.asarray([2.0], dtype=np.float32)}
    peer_divisions = {0: np.asarray([-2.0], dtype=np.float32)}

    scores, divisions = reciprocal_movie_evidence(
        primary_scores,
        primary_divisions,
        peer_scores,
        peer_divisions,
        mode="reciprocal_mean",
    )

    np.testing.assert_array_equal(scores[0], [[0.5, 0.5]])
    np.testing.assert_array_equal(divisions[0], [0.0])


def test_reciprocal_encoding_samples_shared_patches_without_changing_outputs() -> None:
    torch.manual_seed(19)
    primary = PhysicalPatchAssociationModel(base_channels=8, embedding_channels=16)
    peer = PhysicalPatchAssociationModel(base_channels=8, embedding_channels=16)
    video = GraphVideo(
        "fixture",
        node_ids=np.asarray([1, 2]),
        times=np.asarray([0, 1]),
        coords_voxel=np.asarray([[2, 2, 2], [2, 2, 2]], dtype=np.float32),
        edges=np.asarray([[1, 2]], dtype=np.int64),
    )
    images = np.random.default_rng(5).normal(size=(2, 5, 5, 5)).astype(np.float32)
    device = torch.device("cpu")

    first = extract_movie_embeddings(primary, video, images, device)
    second = extract_movie_embeddings(peer, video, images, device)
    shared = extract_reciprocal_movie_embeddings(
        primary, peer, video, images, device
    )

    np.testing.assert_allclose(shared[0], first[0], atol=1e-6)
    np.testing.assert_allclose(shared[1], first[1], atol=1e-6)
    np.testing.assert_allclose(shared[2], second[0], atol=1e-6)
    np.testing.assert_allclose(shared[3], second[1], atol=1e-6)
    assert shared[4]["physical_patch_extractions_per_node"] == 1


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


def test_real_prefix_partition_is_cross_worker_disjoint() -> None:
    paths = [Path(f"44b6_{index:08d}.geff") for index in range(130)]

    validation, calibration, training = real_prefix_partition(
        paths,
        prefix="44b6",
        validation_count=12,
        calibration_count=12,
        training_count=96,
    )
    repeated = real_prefix_partition(
        list(reversed(paths)),
        prefix="44b6",
        validation_count=12,
        calibration_count=12,
        training_count=96,
    )

    assert (validation, calibration, training) == repeated
    assert len(validation) == 12
    assert len(calibration) == 12
    assert len(training) == 96
    assert set(validation).isdisjoint(calibration)
    assert set(validation).isdisjoint(training)
    assert set(calibration).isdisjoint(training)


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
    for ensemble_mode in ENSEMBLE_MODES:
        for appearance_weight in APPEARANCE_WEIGHTS:
            for division_weight in DIVISION_WEIGHTS:
                gain = 0.0 if appearance_weight == 0 else 0.002
                movie_a_gain = -0.001 if appearance_weight == 0.10 else gain
                rows.append(
                    {
                        "ensemble_mode": ensemble_mode,
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
    assert selected["selected_ensemble_mode"] == "target_only"
