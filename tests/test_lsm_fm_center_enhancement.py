from __future__ import annotations

import numpy as np
import pytest
import torch

from research.lsm_fm_detection.center_enhancement import (
    apply_bounded_offsets,
    build_center_enhancement_model,
    center_enhancement_loss,
    extract_center_patches,
    match_annotated_peaks,
    offset_target_distribution,
    spatial_expectation,
)


def test_sparse_annotation_matching_is_one_to_one_and_uses_native_yx_scale() -> None:
    peaks = np.asarray([[2.0, 3.0, 4.0], [2.0, 3.2, 4.1], [9.0, 9.0, 9.0]])
    annotations_native = np.asarray([[2.0, 12.8, 16.8]])
    matched = match_annotated_peaks(peaks, annotations_native)
    assert matched.peak_indices.tolist() == [1]
    np.testing.assert_allclose(matched.offsets_input[0], [0.0, 0.0, 0.1], atol=1e-6)


def test_unmatched_peaks_are_not_returned_as_negative_labels() -> None:
    matched = match_annotated_peaks([[0, 0, 0], [20, 20, 20]], [[0, 0, 0]])
    assert matched.peak_indices.tolist() == [0]
    assert matched.annotation_indices.tolist() == [0]


def test_patch_extraction_is_finite_at_border_and_locally_normalized() -> None:
    raw = np.arange(5 * 5 * 5, dtype=np.float32).reshape(5, 5, 5)
    probability = np.zeros_like(raw)
    probability[0, 0, 0] = 0.7
    patches = extract_center_patches(raw, probability, [[0, 0, 0]], patch_shape=(3, 3, 3))
    assert patches.shape == (1, 2, 3, 3, 3)
    assert np.isfinite(patches).all()
    assert 0.0 <= float(patches.min()) <= float(patches.max()) <= 1.0
    assert patches[0, 1].max() == pytest.approx(1.0)


def test_spatial_expectation_recovers_an_asymmetric_peak() -> None:
    logits = torch.full((1, 1, 5, 5, 5), -20.0)
    logits[0, 0, 3, 1, 2] = 20.0
    offset = spatial_expectation(logits)
    torch.testing.assert_close(offset, torch.tensor([[1.0, -1.0, 0.0]]), atol=1e-5, rtol=0)


def test_target_distribution_and_loss_support_subvoxel_offsets() -> None:
    offsets = torch.tensor([[0.25, -0.40, 0.10]], dtype=torch.float32)
    targets = offset_target_distribution(offsets, patch_shape=(5, 5, 5))
    assert targets.shape == (1, 5, 5, 5)
    torch.testing.assert_close(targets.sum(), torch.tensor(1.0))
    logits = torch.log(targets.clamp_min(1e-8))[:, None].requires_grad_(True)
    result = center_enhancement_loss(logits, offsets)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert logits.grad is not None


def test_model_preserves_batch_and_outputs_bounded_patch_offsets() -> None:
    model = build_center_enhancement_model(channels=8)
    result = model(torch.rand(2, 2, 5, 5, 5))
    assert result["logits"].shape == (2, 1, 5, 5, 5)
    assert result["offsets"].shape == (2, 3)
    assert torch.max(torch.abs(result["offsets"])) <= 2.0


def test_apply_offsets_clips_movement_and_volume_bounds_without_changing_count() -> None:
    refined = apply_bounded_offsets(
        [[0.25, 5.0, 9.75], [4.0, 4.0, 4.0]],
        [[-8.0, 0.5, 8.0], [0.1, -0.2, 0.3]],
        volume_shape=(10, 10, 10),
        maximum_offset_voxels=1.5,
    )
    assert refined.shape == (2, 3)
    np.testing.assert_allclose(refined[0], [0.0, 5.5, 9.0])
    np.testing.assert_allclose(refined[1], [4.1, 3.8, 4.3])
