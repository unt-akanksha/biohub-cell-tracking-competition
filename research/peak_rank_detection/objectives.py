"""Geometry and sparse-supervision objectives for temporal peak detection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _points3(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must be a finite (N, 3) array")
    return values


def points_to_gaussian_heatmap(
    points: np.ndarray | Sequence[Sequence[float]],
    shape: Sequence[int],
    *,
    sigma: float = 1.0,
    truncate: float = 3.0,
) -> np.ndarray:
    spatial_shape = tuple(int(value) for value in shape)
    if len(spatial_shape) != 3 or min(spatial_shape) <= 0:
        raise ValueError("shape must contain three positive dimensions")
    if sigma <= 0 or truncate <= 0:
        raise ValueError("sigma and truncate must be positive")
    result = np.zeros(spatial_shape, dtype=np.float32)
    radius = int(math.ceil(sigma * truncate))
    for point in _points3(points):
        center = np.rint(point).astype(np.int64)
        starts = np.maximum(center - radius, 0)
        stops = np.minimum(center + radius + 1, np.asarray(spatial_shape))
        if np.any(starts >= stops):
            continue
        axes = [
            np.arange(starts[axis], stops[axis], dtype=np.float32)
            for axis in range(3)
        ]
        zz, yy, xx = np.meshgrid(*axes, indexing="ij")
        squared = (
            (zz - point[0]) ** 2
            + (yy - point[1]) ** 2
            + (xx - point[2]) ** 2
        )
        gaussian = np.exp(-0.5 * squared / sigma**2).astype(np.float32)
        slices = tuple(
            slice(int(starts[axis]), int(stops[axis])) for axis in range(3)
        )
        np.maximum(result[slices], gaussian, out=result[slices])
        if np.all((center >= 0) & (center < np.asarray(spatial_shape))):
            result[tuple(center)] = 1.0
    return result


def focal_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CornerNet-style focal loss for max-composed Gaussian peak targets."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if alpha <= 0 or beta <= 0:
        raise ValueError("focal exponents must be positive")
    targets = targets.to(dtype=logits.dtype)
    positive = targets >= 1.0 - 1e-6
    negative = ~positive
    positive_loss = -F.logsigmoid(logits) * torch.sigmoid(-logits).pow(alpha)
    negative_loss = (
        -F.logsigmoid(-logits)
        * torch.sigmoid(logits).pow(alpha)
        * (1.0 - targets).pow(beta)
    )
    positive_count = positive.sum().clamp_min(1)
    return (
        positive_loss[positive].sum() + negative_loss[negative].sum()
    ) / positive_count


def _shell_offsets(
    *, minimum_radius: float, maximum_radius: float, device: torch.device
) -> torch.Tensor:
    extent = int(math.ceil(maximum_radius))
    values = []
    for z in range(-extent, extent + 1):
        for y in range(-extent, extent + 1):
            for x in range(-extent, extent + 1):
                distance = math.sqrt(z * z + y * y + x * x)
                if minimum_radius <= distance <= maximum_radius:
                    values.append((z, y, x))
    if not values:
        raise ValueError("ranking shell contains no offsets")
    return torch.tensor(values, dtype=torch.long, device=device)


def sparse_peak_ranking_loss(
    logits: torch.Tensor,
    points_by_batch: Sequence[torch.Tensor],
    *,
    margin: float = 1.0,
    minimum_radius: float = 2.0,
    maximum_radius: float = 5.0,
    hard_negatives: int = 16,
    exclusion_radius: float = 2.0,
) -> torch.Tensor:
    """Make each annotated peak outrank its hardest safe local neighbors.

    This loss remains valid for positive-unlabeled real crops: it uses only a
    bounded shell around each known cell and excludes voxels near any other
    annotation.  It never treats the rest of a sparsely annotated volume as
    background.
    """

    if logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError("logits must have shape (B, 1, Z, Y, X)")
    if len(points_by_batch) != logits.shape[0]:
        raise ValueError("one point tensor is required per batch element")
    if margin <= 0 or not 0 < minimum_radius <= maximum_radius:
        raise ValueError("invalid ranking margin or shell radii")
    if hard_negatives <= 0 or exclusion_radius < 0:
        raise ValueError("invalid hard-negative configuration")

    shape = torch.tensor(logits.shape[-3:], device=logits.device)
    offsets = _shell_offsets(
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        device=logits.device,
    )
    losses: list[torch.Tensor] = []
    for batch_index, raw_points in enumerate(points_by_batch):
        points = raw_points.to(device=logits.device, dtype=torch.float32).reshape(-1, 3)
        if not len(points):
            continue
        centers = points.round().long()
        inside = ((centers >= 0) & (centers < shape)).all(dim=1)
        centers = centers[inside]
        points = points[inside]
        for point_index, center in enumerate(centers):
            candidates = center[None] + offsets
            valid = ((candidates >= 0) & (candidates < shape)).all(dim=1)
            candidates = candidates[valid]
            if not len(candidates):
                continue
            if len(points) > 1 and exclusion_radius > 0:
                distances = torch.cdist(
                    candidates.float(), points, p=2
                )
                candidates = candidates[
                    (distances >= exclusion_radius).all(dim=1)
                ]
            if not len(candidates):
                continue
            positive_logit = logits[
                batch_index, 0, center[0], center[1], center[2]
            ]
            negative_logits = logits[
                batch_index,
                0,
                candidates[:, 0],
                candidates[:, 1],
                candidates[:, 2],
            ]
            hardest = negative_logits.topk(
                min(hard_negatives, len(negative_logits))
            ).values
            losses.append(F.softplus(margin + hardest - positive_logit).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def subvoxel_offset_loss(
    offsets: torch.Tensor, points_by_batch: Sequence[torch.Tensor]
) -> torch.Tensor:
    if offsets.ndim != 5 or offsets.shape[1] != 3:
        raise ValueError("offsets must have shape (B, 3, Z, Y, X)")
    if len(points_by_batch) != offsets.shape[0]:
        raise ValueError("one point tensor is required per batch element")
    shape = torch.tensor(offsets.shape[-3:], device=offsets.device)
    losses: list[torch.Tensor] = []
    for batch_index, raw_points in enumerate(points_by_batch):
        points = raw_points.to(device=offsets.device, dtype=torch.float32).reshape(-1, 3)
        centers = points.round().long()
        inside = ((centers >= 0) & (centers < shape)).all(dim=1)
        centers, points = centers[inside], points[inside]
        if not len(points):
            continue
        predicted = offsets[
            batch_index, :, centers[:, 0], centers[:, 1], centers[:, 2]
        ].transpose(0, 1)
        losses.append(F.smooth_l1_loss(predicted, points - centers.float(), beta=0.1))
    if not losses:
        return offsets.sum() * 0.0
    return torch.stack(losses).mean()
