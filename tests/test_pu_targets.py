from __future__ import annotations

import numpy as np
import pytest
import torch

from research.spotiflow_biohub.pu_targets import (
    PeakSet,
    YXTransform,
    add_forced_annotations,
    build_pu_targets,
    extract_local_peaks,
    match_teacher_peaks,
    weighted_pu_bce_with_logits,
)


def test_teacher_consensus_is_one_to_one_and_confidence_weighted() -> None:
    primary = PeakSet(
        coords=np.asarray([[2, 2, 2], [2, 2, 4], [8, 8, 8]], dtype=np.float32),
        confidence=np.asarray([0.99, 0.98, 0.99], dtype=np.float32),
    )
    secondary = PeakSet(
        coords=np.asarray([[2, 2, 3], [8, 8, 9]], dtype=np.float32),
        confidence=np.asarray([0.97, 0.99], dtype=np.float32),
    )
    consensus = match_teacher_peaks(primary, secondary, match_radius=1.1)
    assert consensus.shape == (2, 3)
    near_two = consensus[np.argmin(np.linalg.norm(consensus - [2, 2, 3], axis=1))]
    near_eight = consensus[np.argmin(np.linalg.norm(consensus - [8, 8, 9], axis=1))]
    np.testing.assert_allclose(near_two[:2], [2, 2])
    assert 2.0 < near_two[2] < 3.0
    np.testing.assert_allclose(near_eight[:2], [8, 8])


def test_annotations_replace_nearby_consensus_and_add_missed_positive() -> None:
    combined, forced = add_forced_annotations(
        np.asarray([[3, 3, 3], [9, 9, 9]], dtype=np.float32),
        np.asarray([[3, 3, 4], [1, 1, 1]], dtype=np.float32),
        merge_radius=1.1,
    )
    assert forced == 2
    assert {tuple(point) for point in combined} == {
        (9.0, 9.0, 9.0),
        (3.0, 3.0, 4.0),
        (1.0, 1.0, 1.0),
    }


def test_build_targets_keeps_teacher_disagreement_unknown() -> None:
    shape = (12, 12, 12)
    primary = np.zeros(shape, dtype=np.float32)
    secondary = np.zeros(shape, dtype=np.float32)
    primary[4, 4, 4] = 0.99
    secondary[4, 4, 5] = 0.98
    primary[8, 8, 8] = 0.99  # primary-only support must not become background
    secondary[1, 1, 1] = 0.50  # secondary-only low-probability support is unknown

    targets = build_pu_targets(
        primary,
        secondary,
        annotations=[[10, 10, 10]],
        consensus_radius=1.1,
        positive_sigma=0.5,
        positive_weight_floor=0.10,
    )
    assert targets.consensus_count == 1
    assert targets.forced_annotation_count == 1
    assert targets.positive_mask[4, 4, 4] or targets.positive_mask[4, 4, 5]
    assert targets.positive_mask[10, 10, 10]
    assert targets.unknown_mask[8, 8, 8]
    assert targets.unknown_mask[8, 8, 7]  # disagreement support is spatially buffered
    assert targets.unknown_mask[1, 1, 1]
    assert targets.weights[8, 8, 8] == 0
    assert targets.safe_background_mask[0, 11, 0]
    assert targets.weights[0, 11, 0] == pytest.approx(0.02)
    assert not np.any(targets.positive_mask & targets.safe_background_mask)


def test_background_weight_above_guardrail_is_rejected() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="background_weight"):
        build_pu_targets(volume, volume, [], background_weight=0.021)


def test_local_peak_plateau_is_collapsed_deterministically() -> None:
    volume = np.zeros((6, 6, 6), dtype=np.float32)
    volume[2, 2, 2] = 0.99
    volume[2, 2, 3] = 0.99
    peaks = extract_local_peaks(volume, threshold=0.9, min_distance_voxels=1)
    np.testing.assert_array_equal(peaks.coords, [[2, 2, 2]])


@pytest.mark.parametrize(
    "transform",
    [
        YXTransform(flip_y=True),
        YXTransform(flip_x=True),
        YXTransform(rotate_k=1),
        YXTransform(flip_y=True, flip_x=True, rotate_k=3),
    ],
)
def test_yx_augmentation_moves_coordinates_with_voxels(transform: YXTransform) -> None:
    shape = (3, 5, 7)
    point = np.asarray([[1, 2, 4]], dtype=np.float32)
    volume = np.zeros(shape, dtype=np.uint8)
    volume[tuple(point[0].astype(int))] = 1
    transformed_volume = transform.apply_array(volume)
    transformed_points, transformed_shape = transform.apply_coords(point, shape)
    assert transformed_volume.shape == transformed_shape
    index = tuple(np.rint(transformed_points[0]).astype(int))
    assert transformed_volume[index] == 1
    assert transformed_volume.sum() == 1


def test_weighted_loss_ignores_unknown_and_bounds_dense_background() -> None:
    logits = torch.zeros((1, 1, 2, 2, 2), requires_grad=True)
    targets = torch.zeros_like(logits)
    weights = torch.zeros_like(logits)
    targets[..., 0, 0, 0] = 1.0
    weights[..., 0, 0, 0] = 1.0
    weights[..., 1, :, :] = 0.02
    loss = weighted_pu_bce_with_logits(logits, targets, weights)
    assert loss.item() == pytest.approx(np.log(2) * 1.02)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[..., 0, 1, 1].item() == 0.0  # unknown
    assert abs(logits.grad[..., 1, 0, 0].item()) < abs(
        logits.grad[..., 0, 0, 0].item()
    )
