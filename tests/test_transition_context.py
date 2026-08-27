from __future__ import annotations

import numpy as np
import pytest

from research.temporal_contrastive.transition_context import (
    CANDIDATE_CONTEXT_WIDTH,
    candidate_transition_features,
    estimate_transition_context,
)


def structured_volume(shape=(25, 31, 35), seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    volume = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
    volume[4:9, 7:14, 10:18] += 2.0
    volume[15:21, 19:25, 24:30] += 1.0
    return volume


def test_phase_context_recovers_integer_global_shift() -> None:
    reference = structured_volume()
    expected_shift = (3, -4, 5)
    moved = np.roll(reference, expected_shift, axis=(0, 1, 2))

    context = estimate_transition_context(
        reference,
        moved,
        voxel_size_zyx_um=(1.5, 0.5, 0.5),
        max_fft_shape_zyx=reference.shape,
    )

    assert context.global_shift_zyx_voxel == pytest.approx(expected_shift)
    assert context.global_shift_zyx_um == pytest.approx((4.5, -2.0, 2.5))
    assert context.aligned_ncc > 0.999
    assert context.zero_shift_ncc < context.aligned_ncc
    assert context.duplicate_confidence == 0.0
    assert context.reliability > 0.5


def test_exact_repeated_frame_exposes_duplicate_evidence() -> None:
    reference = structured_volume()
    context = estimate_transition_context(
        reference,
        reference.copy(),
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        max_fft_shape_zyx=reference.shape,
    )

    assert context.global_shift_zyx_voxel == (0.0, 0.0, 0.0)
    assert context.zero_shift_ncc == pytest.approx(1.0)
    assert context.aligned_ncc == pytest.approx(1.0)
    assert context.duplicate_confidence == pytest.approx(1.0)


def test_candidate_features_remove_global_shift_and_resist_one_outlier() -> None:
    reference = structured_volume()
    moved = np.roll(reference, (2, -1, 3), axis=(0, 1, 2))
    context = estimate_transition_context(
        reference,
        moved,
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        max_fft_shape_zyx=reference.shape,
    )
    source = np.asarray([[0, 0, 0], [5, 5, 5], [10, 10, 10]], dtype=np.float32)
    target = np.asarray(
        [[2, -1, 3], [7, 4, 8], [12, 9, 13], [30, 30, 30]], dtype=np.float32
    )
    candidates = np.asarray(
        [
            [True, False, False, True],
            [False, True, False, False],
            [False, False, True, False],
        ],
        dtype=bool,
    )

    features = candidate_transition_features(
        source, target, candidates, context, candidate_radius_um=32.0
    )

    assert features.shape == (3, 4, CANDIDATE_CONTEXT_WIDTH)
    assert np.all(features[~candidates] == 0.0)
    for row, column in ((0, 0), (1, 1), (2, 2)):
        assert np.linalg.vector_norm(features[row, column, :3]) < 1e-7
        assert features[row, column, 3] < 1e-7
        assert features[row, column, 7] < features[0, 3, 7]
    assert np.isfinite(features[candidates]).all()


def test_context_bounds_fft_and_rejects_invalid_inputs() -> None:
    reference = structured_volume(shape=(33, 65, 67))
    moved = np.roll(reference, (2, 4, -6), axis=(0, 1, 2))
    context = estimate_transition_context(
        reference,
        moved,
        voxel_size_zyx_um=(1.0, 0.5, 0.5),
        max_fft_shape_zyx=(16, 32, 32),
    )
    assert context.fft_stride_zyx == (3, 3, 3)
    assert np.isfinite(context.feature_vector(candidate_radius_um=32.0)).all()

    with pytest.raises(ValueError, match="equal shape"):
        estimate_transition_context(
            reference,
            moved[:-1],
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="nonconstant"):
        estimate_transition_context(
            np.ones((5, 5, 5)),
            np.ones((5, 5, 5)),
            voxel_size_zyx_um=(1.0, 1.0, 1.0),
        )
