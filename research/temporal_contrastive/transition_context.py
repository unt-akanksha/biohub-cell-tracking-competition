"""Label-free transition and candidate-motion context for Biohub v3.

The implementation is project-authored and deliberately independent of the
learned appearance models.  It estimates a bounded whole-volume translation
from the two images, exposes duplicate-frame evidence without making a hard
tracking decision, and represents candidate motion after subtracting that
global translation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


DEFAULT_MAX_FFT_SHAPE_ZYX = (64, 128, 128)
DEFAULT_CANDIDATE_RADIUS_UM = 32.0
DUPLICATE_NCC_FLOOR = 0.995
CANDIDATE_CONTEXT_WIDTH = 18


def _positive_triplet(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(f"{name} must contain three positive finite values")
    return result


def _finite_volume(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 3 or min(result.shape) < 3:
        raise ValueError(f"{name} must be a three-dimensional volume")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _normalized_flat(value: np.ndarray) -> np.ndarray:
    centered = np.asarray(value, dtype=np.float64).reshape(-1)
    centered = centered - centered.mean()
    norm = np.linalg.vector_norm(centered)
    if norm <= 1e-12:
        return np.zeros_like(centered)
    return centered / norm


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = _normalized_flat(first)
    right = _normalized_flat(second)
    if not np.any(left) or not np.any(right):
        return 0.0
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _overlap_slices(
    shape: Sequence[int], shift_zyx: Sequence[int]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    reference_slices: list[slice] = []
    moved_slices: list[slice] = []
    for size, shift in zip(shape, shift_zyx, strict=True):
        offset = int(shift)
        if abs(offset) >= int(size):
            raise ValueError("estimated shift has no image overlap")
        if offset >= 0:
            reference_slices.append(slice(0, int(size) - offset))
            moved_slices.append(slice(offset, int(size)))
        else:
            reference_slices.append(slice(-offset, int(size)))
            moved_slices.append(slice(0, int(size) + offset))
    return tuple(reference_slices), tuple(moved_slices)


def _bounded_view(
    volume: np.ndarray, max_shape_zyx: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    maximum = _positive_triplet(max_shape_zyx, "maximum FFT shape").astype(int)
    stride = np.maximum(1, np.ceil(np.asarray(volume.shape) / maximum).astype(int))
    view = volume[:: stride[0], :: stride[1], :: stride[2]]
    return view, stride


def _second_peak_outside_neighborhood(
    correlation: np.ndarray, peak_index: Sequence[int]
) -> float:
    masked = np.array(correlation, dtype=np.float64, copy=True)
    shape = masked.shape
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                index = tuple(
                    (int(center) + delta) % int(size)
                    for center, delta, size in zip(
                        peak_index, (dz, dy, dx), shape, strict=True
                    )
                )
                masked[index] = -np.inf
    finite = masked[np.isfinite(masked)]
    return float(np.max(finite)) if finite.size else 0.0


@dataclass(frozen=True)
class TransitionContext:
    """Immutable image-derived evidence for one adjacent-frame transition."""

    global_shift_zyx_voxel: tuple[float, float, float]
    global_shift_zyx_um: tuple[float, float, float]
    zero_shift_ncc: float
    aligned_ncc: float
    phase_peak_margin: float
    duplicate_confidence: float
    reliability: float
    fft_stride_zyx: tuple[int, int, int]

    def feature_vector(self, *, candidate_radius_um: float) -> np.ndarray:
        radius = float(candidate_radius_um)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("candidate radius must be positive and finite")
        return np.asarray(
            (
                *(np.asarray(self.global_shift_zyx_um, dtype=np.float64) / radius),
                self.zero_shift_ncc,
                self.aligned_ncc,
                self.phase_peak_margin,
                self.duplicate_confidence,
                self.reliability,
            ),
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_transition_context(
    reference_volume: np.ndarray,
    moved_volume: np.ndarray,
    *,
    voxel_size_zyx_um: Sequence[float],
    max_fft_shape_zyx: Sequence[int] = DEFAULT_MAX_FFT_SHAPE_ZYX,
) -> TransitionContext:
    """Estimate label-free translation and transition quality from two images.

    ``global_shift_zyx_voxel`` maps coordinates in ``reference_volume`` to
    coordinates in ``moved_volume``.  The FFT view is strided to bound memory;
    the reported shift is scaled back to original voxel coordinates.
    """

    reference = _finite_volume(reference_volume, "reference volume")
    moved = _finite_volume(moved_volume, "moved volume")
    if reference.shape != moved.shape:
        raise ValueError("transition volumes must have equal shape")
    voxel_size = _positive_triplet(voxel_size_zyx_um, "voxel size")
    reference_view, stride = _bounded_view(reference, max_fft_shape_zyx)
    moved_view = moved[:: stride[0], :: stride[1], :: stride[2]]

    first = reference_view.astype(np.float64) - float(reference_view.mean())
    second = moved_view.astype(np.float64) - float(moved_view.mean())
    if np.linalg.vector_norm(first) <= 1e-12 or np.linalg.vector_norm(second) <= 1e-12:
        raise ValueError("transition context requires nonconstant image volumes")
    first_fft = np.fft.fftn(first)
    second_fft = np.fft.fftn(second)
    cross_power = second_fft * np.conj(first_fft)
    magnitude = np.abs(cross_power)
    cross_power /= np.maximum(magnitude, 1e-12)
    correlation = np.fft.ifftn(cross_power).real
    peak_index = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift_view = np.asarray(
        [
            index - size if index > size // 2 else index
            for index, size in zip(peak_index, correlation.shape, strict=True)
        ],
        dtype=np.int64,
    )
    shift_voxel = shift_view * stride
    reference_slices, moved_slices = _overlap_slices(reference.shape, shift_voxel)
    zero_ncc = _normalized_correlation(reference, moved)
    aligned_ncc = _normalized_correlation(
        reference[reference_slices], moved[moved_slices]
    )
    peak = float(correlation[peak_index])
    second_peak = _second_peak_outside_neighborhood(correlation, peak_index)
    peak_margin = float(
        np.clip((peak - second_peak) / max(abs(peak), 1e-12), 0.0, 1.0)
    )
    shift_norm = float(np.linalg.vector_norm(shift_voxel))
    duplicate_similarity = np.clip(
        (zero_ncc - DUPLICATE_NCC_FLOOR) / (1.0 - DUPLICATE_NCC_FLOOR),
        0.0,
        1.0,
    )
    duplicate_confidence = float(duplicate_similarity * np.exp(-shift_norm))
    reliability = float(
        np.clip(max(aligned_ncc, 0.0) * (0.5 + 0.5 * peak_margin), 0.0, 1.0)
    )
    shift_um = shift_voxel.astype(np.float64) * voxel_size
    return TransitionContext(
        global_shift_zyx_voxel=tuple(float(value) for value in shift_voxel),
        global_shift_zyx_um=tuple(float(value) for value in shift_um),
        zero_shift_ncc=zero_ncc,
        aligned_ncc=aligned_ncc,
        phase_peak_margin=peak_margin,
        duplicate_confidence=duplicate_confidence,
        reliability=reliability,
        fft_stride_zyx=tuple(int(value) for value in stride),
    )


def candidate_transition_features(
    source_coords_zyx_um: np.ndarray,
    target_coords_zyx_um: np.ndarray,
    candidate_mask: np.ndarray,
    context: TransitionContext,
    *,
    candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
) -> np.ndarray:
    """Build dense v3 edge context while evaluating eligible pairs only.

    Ineligible entries are exactly zero.  Eligible features contain residual
    motion after global-shift subtraction, robust transition-consensus
    deviation, candidate-density summaries, and the immutable image token.
    """

    source = np.asarray(source_coords_zyx_um, dtype=np.float64)
    target = np.asarray(target_coords_zyx_um, dtype=np.float64)
    candidates = np.asarray(candidate_mask)
    radius = float(candidate_radius_um)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("source coordinates must have shape (N, 3)")
    if target.ndim != 2 or target.shape[1:] != (3,):
        raise ValueError("target coordinates must have shape (M, 3)")
    if candidates.shape != (len(source), len(target)) or candidates.dtype != np.bool_:
        raise ValueError("candidate mask must be boolean with shape (N, M)")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("candidate coordinates must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("candidate radius must be positive and finite")
    output = np.zeros(
        (len(source), len(target), CANDIDATE_CONTEXT_WIDTH), dtype=np.float32
    )
    pair_indices = np.argwhere(candidates)
    if not len(pair_indices):
        return output

    source_rows = pair_indices[:, 0]
    target_rows = pair_indices[:, 1]
    shift = np.asarray(context.global_shift_zyx_um, dtype=np.float64)
    residual = target[target_rows] - source[source_rows] - shift[None]
    consensus = np.median(residual, axis=0)
    mad = 1.4826 * np.median(np.abs(residual - consensus[None]), axis=0)
    robust_scale = np.maximum(mad, radius * 0.02)
    standardized = np.clip(
        (residual - consensus[None]) / robust_scale[None], -8.0, 8.0
    )
    consensus_distance = np.linalg.vector_norm(standardized, axis=1, keepdims=True)
    source_counts = candidates.sum(axis=1)
    target_counts = candidates.sum(axis=0)
    source_density = np.log1p(source_counts[source_rows]) / np.log1p(max(len(target), 1))
    target_density = np.log1p(target_counts[target_rows]) / np.log1p(max(len(source), 1))
    transition_token = context.feature_vector(candidate_radius_um=radius)
    features = np.concatenate(
        (
            residual / radius,
            np.linalg.vector_norm(residual, axis=1, keepdims=True) / radius,
            standardized,
            consensus_distance,
            source_density[:, None],
            target_density[:, None],
            np.repeat(transition_token[None], len(pair_indices), axis=0),
        ),
        axis=1,
    )
    if features.shape[1] != CANDIDATE_CONTEXT_WIDTH or not np.isfinite(features).all():
        raise RuntimeError("candidate transition feature construction changed")
    output[source_rows, target_rows] = features.astype(np.float32)
    return output
