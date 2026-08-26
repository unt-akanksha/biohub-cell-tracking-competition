"""Positive-unlabeled targets for dense 3D Biohub detector adaptation.

The Biohub annotations contain reliable positives but do not identify true
background.  This module therefore keeps three distinct regions:

* consensus teacher peaks and organizer annotations are positive;
* either teacher's low-confidence support is unknown; and
* only voxels outside both supports receive a small background weight.

The implementation is independent of Spotiflow so the target geometry and
loss masking can be tested before spending GPU quota on model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PeakSet:
    """Coordinates and confidence values for one teacher's local maxima."""

    coords: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class PUTargets:
    """Dense target and supervision masks for one 3D training crop."""

    heatmap: np.ndarray
    weights: np.ndarray
    positive_mask: np.ndarray
    safe_background_mask: np.ndarray
    unknown_mask: np.ndarray
    positive_coords: np.ndarray
    consensus_count: int
    forced_annotation_count: int


@dataclass(frozen=True)
class YXTransform:
    """Coordinate-consistent flips followed by a 90-degree Y/X rotation."""

    flip_y: bool = False
    flip_x: bool = False
    rotate_k: int = 0

    def apply_array(self, array: np.ndarray) -> np.ndarray:
        if array.ndim < 3:
            raise ValueError("array must end in three spatial dimensions")
        result = array
        if self.flip_y:
            result = np.flip(result, axis=-2)
        if self.flip_x:
            result = np.flip(result, axis=-1)
        return np.rot90(result, k=self.rotate_k % 4, axes=(-2, -1)).copy()

    def apply_coords(
        self, coords: np.ndarray, spatial_shape: Sequence[int]
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        shape = _shape3(spatial_shape)
        points = _coords3(coords).copy()
        _, height, width = shape
        if self.flip_y:
            points[:, 1] = height - 1 - points[:, 1]
        if self.flip_x:
            points[:, 2] = width - 1 - points[:, 2]
        for _ in range(self.rotate_k % 4):
            y = points[:, 1].copy()
            x = points[:, 2].copy()
            points[:, 1] = width - 1 - x
            points[:, 2] = y
            height, width = width, height
        return points, (shape[0], height, width)


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(value) for value in shape)
    if len(result) != 3 or any(value <= 0 for value in result):
        raise ValueError(f"expected a positive 3D shape, got {result}")
    return result


def _coords3(coords: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    result = np.asarray(coords, dtype=np.float32)
    if result.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("coordinates must be a finite (N, 3) array")
    return result


def _probability_volume(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3D probability volume")
    if result.size and (float(result.min()) < 0.0 or float(result.max()) > 1.0):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return result


def extract_local_peaks(
    probabilities: np.ndarray,
    *,
    threshold: float,
    min_distance_voxels: int = 1,
) -> PeakSet:
    """Extract deterministic local maxima, collapsing flat plateaus to one voxel."""

    volume = _probability_volume(probabilities, "probabilities")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    if min_distance_voxels < 0:
        raise ValueError("min_distance_voxels must be nonnegative")
    size = 2 * int(min_distance_voxels) + 1
    maxima = maximum_filter(volume, size=size, mode="constant", cval=-np.inf)
    candidates = np.argwhere((volume >= threshold) & (volume == maxima))
    if len(candidates) == 0:
        return PeakSet(
            coords=np.empty((0, 3), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
        )

    # Equal-valued plateaus can contain adjacent candidates.  Confidence-first
    # non-maximum suppression makes the representative deterministic.
    confidence = volume[tuple(candidates.T)]
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-float(confidence[i]), *map(int, candidates[i])),
    )
    tree = cKDTree(candidates.astype(np.float32))
    suppressed: set[int] = set()
    kept: list[int] = []
    radius = max(float(min_distance_voxels), 0.5)
    for index in order:
        if index in suppressed:
            continue
        kept.append(index)
        suppressed.update(tree.query_ball_point(candidates[index], r=radius))
    kept_array = np.asarray(kept, dtype=np.int64)
    return PeakSet(
        coords=candidates[kept_array].astype(np.float32),
        confidence=confidence[kept_array].astype(np.float32),
    )


def match_teacher_peaks(
    primary: PeakSet,
    secondary: PeakSet,
    *,
    match_radius: float,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return confidence-weighted one-to-one consensus coordinates.

    Candidate pairs are considered nearest-first.  Each teacher peak may
    support at most one consensus target, preventing a dense cluster from
    multiplying pseudo-labels.
    """

    if match_radius <= 0:
        raise ValueError("match_radius must be positive")
    spacing = np.asarray(voxel_size, dtype=np.float32)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("voxel_size must contain three positive finite values")
    p_coords = _coords3(primary.coords)
    s_coords = _coords3(secondary.coords)
    p_conf = np.asarray(primary.confidence, dtype=np.float32)
    s_conf = np.asarray(secondary.confidence, dtype=np.float32)
    if p_conf.shape != (len(p_coords),) or s_conf.shape != (len(s_coords),):
        raise ValueError("peak confidence length does not match coordinates")
    if len(p_coords) == 0 or len(s_coords) == 0:
        return np.empty((0, 3), dtype=np.float32)

    secondary_tree = cKDTree(s_coords * spacing)
    pairs: list[tuple[float, float, int, int]] = []
    for p_index, coord in enumerate(p_coords * spacing):
        for s_index in secondary_tree.query_ball_point(coord, r=float(match_radius)):
            distance = float(np.linalg.norm(coord - s_coords[s_index] * spacing))
            support = float(min(p_conf[p_index], s_conf[s_index]))
            pairs.append((distance, -support, p_index, int(s_index)))
    pairs.sort()

    used_primary: set[int] = set()
    used_secondary: set[int] = set()
    consensus: list[np.ndarray] = []
    for _, _, p_index, s_index in pairs:
        if p_index in used_primary or s_index in used_secondary:
            continue
        used_primary.add(p_index)
        used_secondary.add(s_index)
        p_weight = float(p_conf[p_index])
        s_weight = float(s_conf[s_index])
        consensus.append(
            (p_weight * p_coords[p_index] + s_weight * s_coords[s_index])
            / max(p_weight + s_weight, 1e-8)
        )
    return (
        np.asarray(consensus, dtype=np.float32).reshape(-1, 3)
        if consensus
        else np.empty((0, 3), dtype=np.float32)
    )


def add_forced_annotations(
    consensus: np.ndarray,
    annotations: np.ndarray,
    *,
    merge_radius: float,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, int]:
    """Merge annotations into pseudo-labels, with annotations taking precedence."""

    if merge_radius < 0:
        raise ValueError("merge_radius must be nonnegative")
    consensus_coords = _coords3(consensus)
    annotation_coords = _coords3(annotations)
    spacing = np.asarray(voxel_size, dtype=np.float32)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.isfinite(spacing).all():
        raise ValueError("voxel_size must contain three positive finite values")

    kept_consensus = consensus_coords
    if len(consensus_coords) and len(annotation_coords):
        annotation_tree = cKDTree(annotation_coords * spacing)
        keep = [
            not annotation_tree.query_ball_point(coord * spacing, r=merge_radius)
            for coord in consensus_coords
        ]
        kept_consensus = consensus_coords[np.asarray(keep, dtype=bool)]
    combined = np.concatenate([kept_consensus, annotation_coords], axis=0)
    return combined.astype(np.float32, copy=False), len(annotation_coords)


def points_to_gaussian_heatmap(
    coords: np.ndarray,
    shape: Sequence[int],
    *,
    sigma: float = 1.0,
    truncate: float = 3.0,
) -> np.ndarray:
    """Rasterize points as a max-composed Gaussian heatmap."""

    spatial_shape = _shape3(shape)
    points = _coords3(coords)
    if sigma <= 0 or truncate <= 0:
        raise ValueError("sigma and truncate must be positive")
    heatmap = np.zeros(spatial_shape, dtype=np.float32)
    radius = int(np.ceil(sigma * truncate))
    for point in points:
        center = np.rint(point).astype(np.int64)
        starts = np.maximum(center - radius, 0)
        stops = np.minimum(center + radius + 1, np.asarray(spatial_shape))
        if np.any(starts >= stops):
            continue
        axes = [np.arange(starts[i], stops[i], dtype=np.float32) for i in range(3)]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        distance_sq = (
            (zz - point[0]) ** 2 + (yy - point[1]) ** 2 + (xx - point[2]) ** 2
        )
        gaussian = np.exp(-0.5 * distance_sq / (sigma**2)).astype(np.float32)
        slices = tuple(slice(int(starts[i]), int(stops[i])) for i in range(3))
        np.maximum(heatmap[slices], gaussian, out=heatmap[slices])
    return heatmap


def build_pu_targets(
    primary_probabilities: np.ndarray,
    secondary_probabilities: np.ndarray,
    annotations: np.ndarray | Sequence[Sequence[float]],
    *,
    high_threshold: float = 0.96875,
    low_support_threshold: float = 0.10,
    peak_min_distance_voxels: int = 1,
    consensus_radius: float = 2.5,
    annotation_merge_radius: float = 2.5,
    positive_sigma: float = 1.0,
    positive_weight_floor: float = 0.05,
    support_dilation_voxels: int = 2,
    background_weight: float = 0.02,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
) -> PUTargets:
    """Build conservative dense supervision from two teachers and known positives."""

    primary_volume = _probability_volume(primary_probabilities, "primary_probabilities")
    secondary_volume = _probability_volume(
        secondary_probabilities, "secondary_probabilities"
    )
    if primary_volume.shape != secondary_volume.shape:
        raise ValueError("teacher probability volumes must have identical shapes")
    if not 0.0 <= low_support_threshold < high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
    if not 0.0 < positive_weight_floor < 1.0:
        raise ValueError("positive_weight_floor must lie in (0, 1)")
    if support_dilation_voxels < 0:
        raise ValueError("support_dilation_voxels must be nonnegative")
    if not 0.0 < background_weight <= 0.02:
        raise ValueError("background_weight must lie in (0, 0.02]")

    primary_peaks = extract_local_peaks(
        primary_volume,
        threshold=high_threshold,
        min_distance_voxels=peak_min_distance_voxels,
    )
    secondary_peaks = extract_local_peaks(
        secondary_volume,
        threshold=high_threshold,
        min_distance_voxels=peak_min_distance_voxels,
    )
    consensus = match_teacher_peaks(
        primary_peaks,
        secondary_peaks,
        match_radius=consensus_radius,
        voxel_size=voxel_size,
    )
    positives, forced_count = add_forced_annotations(
        consensus,
        _coords3(annotations),
        merge_radius=annotation_merge_radius,
        voxel_size=voxel_size,
    )
    heatmap = points_to_gaussian_heatmap(
        positives, primary_volume.shape, sigma=positive_sigma
    )
    positive_mask = heatmap >= positive_weight_floor
    teacher_support = (
        (primary_volume >= low_support_threshold)
        | (secondary_volume >= low_support_threshold)
    )
    if support_dilation_voxels:
        teacher_support = maximum_filter(
            teacher_support,
            size=2 * support_dilation_voxels + 1,
            mode="constant",
            cval=False,
        )
    outside_teacher_support = ~teacher_support
    safe_background_mask = outside_teacher_support & ~positive_mask
    weights = np.zeros(primary_volume.shape, dtype=np.float32)
    weights[positive_mask] = heatmap[positive_mask]
    weights[safe_background_mask] = np.float32(background_weight)
    unknown_mask = ~(positive_mask | safe_background_mask)

    if np.any(positive_mask & safe_background_mask) or np.any(weights[unknown_mask] != 0):
        raise RuntimeError("positive, background, and unknown supervision regions overlap")
    return PUTargets(
        heatmap=heatmap,
        weights=weights,
        positive_mask=positive_mask,
        safe_background_mask=safe_background_mask,
        unknown_mask=unknown_mask,
        positive_coords=positives,
        consensus_count=len(consensus),
        forced_annotation_count=forced_count,
    )


def weighted_pu_bce_with_logits(logits, targets, weights):
    """Torch loss with separate normalization for positives and background.

    The function imports Torch lazily, keeping target construction usable in a
    NumPy-only preprocessing process.  The background term can never dominate
    merely because there are many more background voxels.
    """

    import torch
    import torch.nn.functional as torch_functional

    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError("logits, targets, and weights must have identical shapes")
    raw = torch_functional.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    positive = (targets > 0) & (weights > 0)
    background = (targets == 0) & (weights > 0)
    zero = logits.sum() * 0.0
    positive_loss = (
        (raw[positive] * weights[positive]).sum() / weights[positive].sum().clamp_min(1e-8)
        if torch.any(positive)
        else zero
    )
    background_loss = (
        (raw[background] * weights[background]).sum()
        / weights[background].sum().clamp_min(1e-8)
        if torch.any(background)
        else zero
    )
    # The dense background contribution remains bounded by the configured
    # per-voxel background weight after count normalization.
    background_scale = weights[background].max() if torch.any(background) else zero
    return positive_loss + background_scale * background_loss
