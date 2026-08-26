"""Inference-only sub-voxel refiners for dense 3D cell-center heatmaps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _volume(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3D volume")
    return result


def _points(coords: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(coords, dtype=np.float32)
    if result.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("coords must be a finite (N, 3) array")
    return result


def refine_peaks_weighted(
    probability: np.ndarray,
    coords: np.ndarray | Sequence[Sequence[float]],
    *,
    intensity: np.ndarray | None = None,
    radius: int = 1,
    probability_power: float = 2.0,
    intensity_power: float = 0.0,
    background_quantile: float = 0.25,
) -> np.ndarray:
    """Refine maxima with probability and optional raw-intensity evidence.

    Raw intensity is locally background-subtracted and normalized, so a global
    brightness shift cannot move a peak. Setting ``intensity_power`` to zero is
    the probability-only control used by the current detector.
    """

    heatmap = _volume(probability, "probability")
    points = _points(coords)
    raw = None if intensity is None else _volume(intensity, "intensity")
    if raw is not None and raw.shape != heatmap.shape:
        raise ValueError("intensity and probability shapes must match")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if probability_power <= 0 or intensity_power < 0:
        raise ValueError("probability power must be positive and intensity power nonnegative")
    if not 0.0 <= background_quantile < 1.0:
        raise ValueError("background_quantile must lie in [0, 1)")
    if intensity_power and raw is None:
        raise ValueError("positive intensity_power requires an intensity volume")
    if radius == 0 or not len(points):
        return points.copy()

    refined = np.empty_like(points)
    shape = np.asarray(heatmap.shape, dtype=np.int64)
    for index, point in enumerate(points):
        center = np.rint(point).astype(np.int64)
        starts = np.maximum(center - radius, 0)
        stops = np.minimum(center + radius + 1, shape)
        slices = tuple(slice(int(starts[axis]), int(stops[axis])) for axis in range(3))
        probability_window = np.clip(heatmap[slices], 0.0, None)
        weights = np.power(probability_window, probability_power, dtype=np.float32)
        if intensity_power:
            intensity_window = raw[slices]
            background = float(np.quantile(intensity_window, background_quantile))
            intensity_support = np.clip(intensity_window - background, 0.0, None)
            scale = float(intensity_support.max())
            if scale > 1e-8:
                intensity_support = intensity_support / scale
                weights *= np.power(intensity_support, intensity_power, dtype=np.float32)
        total = float(weights.sum())
        if total <= 1e-8:
            refined[index] = point
            continue
        axes = [
            np.arange(starts[axis], stops[axis], dtype=np.float32)
            for axis in range(3)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        refined[index] = np.asarray(
            [float((grid * weights).sum()) / total for grid in grids],
            dtype=np.float32,
        )
    return refined


def refine_peaks_quadratic(
    probability: np.ndarray,
    coords: np.ndarray | Sequence[Sequence[float]],
    *,
    maximum_offset: float = 1.25,
) -> np.ndarray:
    """Fit a concave 3D quadratic in each 3×3×3 peak neighborhood.

    Fits with a non-negative Hessian eigenvalue, a singular Hessian, or a
    vertex outside the local neighborhood fall back to the input coordinate.
    """

    heatmap = _volume(probability, "probability")
    points = _points(coords)
    if maximum_offset <= 0:
        raise ValueError("maximum_offset must be positive")
    refined = points.copy()
    shape = np.asarray(heatmap.shape, dtype=np.int64)
    offsets = np.asarray(
        [(z, y, x) for z in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )
    design = np.column_stack(
        [
            np.ones(len(offsets)),
            offsets,
            offsets[:, 0] ** 2,
            offsets[:, 1] ** 2,
            offsets[:, 2] ** 2,
            offsets[:, 0] * offsets[:, 1],
            offsets[:, 0] * offsets[:, 2],
            offsets[:, 1] * offsets[:, 2],
        ]
    )
    for index, point in enumerate(points):
        center = np.rint(point).astype(np.int64)
        if np.any(center <= 0) or np.any(center >= shape - 1):
            continue
        window = heatmap[
            center[0] - 1 : center[0] + 2,
            center[1] - 1 : center[1] + 2,
            center[2] - 1 : center[2] + 2,
        ].astype(np.float64)
        coefficients, *_ = np.linalg.lstsq(design, window.reshape(-1), rcond=None)
        gradient = coefficients[1:4]
        zz, yy, xx, zy, zx, yx = coefficients[4:]
        hessian = np.asarray(
            [[2.0 * zz, zy, zx], [zy, 2.0 * yy, yx], [zx, yx, 2.0 * xx]],
            dtype=np.float64,
        )
        if np.max(np.linalg.eigvalsh(hessian)) >= -1e-8:
            continue
        try:
            delta = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(delta).all() or np.max(np.abs(delta)) > maximum_offset:
            continue
        refined[index] = center.astype(np.float32) + delta.astype(np.float32)
    return refined

