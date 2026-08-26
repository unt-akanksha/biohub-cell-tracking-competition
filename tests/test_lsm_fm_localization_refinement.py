from __future__ import annotations

import numpy as np
import pytest

from research.lsm_fm_detection.localization_refinement import (
    refine_peaks_log_quadratic,
    refine_peaks_quadratic,
    refine_peaks_weighted,
)


def gaussian(shape: tuple[int, int, int], center: np.ndarray, sigma: float) -> np.ndarray:
    axes = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in shape],
        indexing="ij",
    )
    squared = sum((axis - center[index]) ** 2 for index, axis in enumerate(axes))
    return np.exp(-0.5 * squared / sigma**2).astype(np.float32)


def test_weighted_refinement_recovers_subvoxel_center() -> None:
    center = np.asarray([3.35, 4.20, 5.65], dtype=np.float32)
    probability = gaussian((9, 10, 11), center, sigma=1.0)
    integer = np.rint(center)[None]
    refined = refine_peaks_weighted(
        probability,
        integer,
        radius=2,
        probability_power=2.0,
    )
    assert np.linalg.norm(refined[0] - center) < np.linalg.norm(integer[0] - center)
    assert np.linalg.norm(refined[0] - center) < 0.08


def test_joint_refinement_uses_locally_background_subtracted_intensity() -> None:
    probability_center = np.asarray([4.0, 4.0, 4.0], dtype=np.float32)
    intensity_center = np.asarray([4.0, 4.45, 4.35], dtype=np.float32)
    probability = gaussian((9, 9, 9), probability_center, sigma=1.1)
    intensity = 7.0 + 3.0 * gaussian((9, 9, 9), intensity_center, sigma=0.8)
    control = refine_peaks_weighted(probability, [[4, 4, 4]], radius=2)
    joint = refine_peaks_weighted(
        probability,
        [[4, 4, 4]],
        intensity=intensity,
        radius=2,
        probability_power=2.0,
        intensity_power=1.0,
    )
    assert np.linalg.norm(joint[0] - intensity_center) < np.linalg.norm(
        control[0] - intensity_center
    )


def test_quadratic_refinement_recovers_concave_gaussian_vertex() -> None:
    center = np.asarray([4.25, 4.10, 3.70], dtype=np.float32)
    probability = gaussian((9, 9, 9), center, sigma=1.4)
    integer = np.rint(center)[None]
    refined = refine_peaks_quadratic(probability, integer)
    assert np.linalg.norm(refined[0] - center) < np.linalg.norm(integer[0] - center)


def test_log_quadratic_is_exact_for_an_unclipped_gaussian_peak() -> None:
    center = np.asarray([4.31, 3.84, 4.22], dtype=np.float32)
    probability = 0.8 * gaussian((9, 9, 9), center, sigma=1.1)
    integer = np.rint(center)[None]
    direct = refine_peaks_quadratic(probability, integer)
    refined = refine_peaks_log_quadratic(probability, integer)
    assert np.linalg.norm(refined[0] - center) < 1e-4
    assert np.linalg.norm(refined[0] - center) < np.linalg.norm(direct[0] - center)


def test_invalid_joint_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires an intensity"):
        refine_peaks_weighted(np.ones((3, 3, 3)), [[1, 1, 1]], intensity_power=1.0)
