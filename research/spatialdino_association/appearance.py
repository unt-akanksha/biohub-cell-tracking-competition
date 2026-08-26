from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _tuple3(values: Sequence[int], name: str) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain three positive integers")
    return result[0], result[1], result[2]


def _feature_indices(
    coords_zyx: torch.Tensor,
    *,
    patch_size_zyx: tuple[int, int, int],
    grid_shape_zyx: tuple[int, int, int],
) -> torch.Tensor:
    """Map input voxel centers onto non-overlapping patch-token centers."""
    patch = coords_zyx.new_tensor(patch_size_zyx)
    indices = (coords_zyx + 0.5) / patch - 0.5
    upper = coords_zyx.new_tensor(grid_shape_zyx) - 1
    return torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), upper)


def _normalized_grid(indices_zyx: torch.Tensor, grid_shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    axes = []
    for axis, size in enumerate(grid_shape_zyx):
        if size == 1:
            axes.append(torch.zeros_like(indices_zyx[:, axis]))
        else:
            axes.append(2.0 * indices_zyx[:, axis] / float(size - 1) - 1.0)
    # grid_sample expects x, y, z in the last dimension.
    return torch.stack((axes[2], axes[1], axes[0]), dim=1)


def sample_patch_embeddings(
    feature_grid: np.ndarray | torch.Tensor,
    coords_zyx: np.ndarray | torch.Tensor,
    *,
    input_shape_zyx: Sequence[int],
    patch_size_zyx: Sequence[int] = (8, 8, 8),
    patch_channels: int = 384,
    normalize: bool = True,
) -> np.ndarray:
    """Trilinearly sample frozen patch tokens at Biohub node coordinates.

    SpatialDINO emits ``[C, Zp, Yp, Xp]`` grids. Its first 384 channels are
    patch embeddings and the remaining channels are attention maps. Coordinates
    must already be expressed in the encoder input grid; for isotropic Biohub
    inference this means multiplying raw z by four before calling this function.
    """
    input_shape = _tuple3(input_shape_zyx, "input_shape_zyx")
    patch_size = _tuple3(patch_size_zyx, "patch_size_zyx")
    features = torch.as_tensor(feature_grid, dtype=torch.float32)
    coords = torch.as_tensor(coords_zyx, dtype=torch.float32)
    if features.ndim != 4:
        raise ValueError("feature_grid must have shape (C, Zp, Yp, Xp)")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_zyx must have shape (N, 3)")
    if patch_channels <= 0 or patch_channels > features.shape[0]:
        raise ValueError("patch_channels is outside the feature grid")
    if not torch.isfinite(features).all() or not torch.isfinite(coords).all():
        raise ValueError("features and coordinates must be finite")
    if any(input_shape[axis] < patch_size[axis] for axis in range(3)):
        raise ValueError("input shape must be at least one patch on every axis")

    grid_shape = tuple(int(value) for value in features.shape[1:])
    expected_grid = tuple(input_shape[axis] // patch_size[axis] for axis in range(3))
    if grid_shape != expected_grid:
        raise ValueError(
            f"feature grid {grid_shape} does not match input/patch grid {expected_grid}"
        )
    if len(coords) == 0:
        return np.empty((0, patch_channels), dtype=np.float32)
    lower_ok = torch.all(coords >= 0)
    upper = coords.new_tensor(input_shape) - 1
    if not bool(lower_ok and torch.all(coords <= upper)):
        raise ValueError("node coordinates are outside the encoder input volume")

    indices = _feature_indices(
        coords,
        patch_size_zyx=patch_size,
        grid_shape_zyx=grid_shape,
    )
    grid = _normalized_grid(indices, grid_shape).reshape(1, 1, 1, len(coords), 3)
    sampled = F.grid_sample(
        features[:patch_channels].unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, 0, 0].transpose(0, 1)
    if normalize:
        sampled = F.normalize(sampled, p=2, dim=1, eps=1e-8)
    return sampled.cpu().numpy().astype(np.float32, copy=False)


def edge_cosine_scores(
    node_embeddings: np.ndarray,
    edge_indices: np.ndarray,
) -> np.ndarray:
    """Return bounded source-target cosine scores for candidate edges."""
    embeddings = np.asarray(node_embeddings, dtype=np.float32)
    edges = np.asarray(edge_indices, dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError("node_embeddings must have shape (N, C)")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edge_indices must have shape (E, 2)")
    if not np.isfinite(embeddings).all():
        raise ValueError("node_embeddings must be finite")
    if len(edges) == 0:
        return np.empty(0, dtype=np.float32)
    if edges.min() < 0 or edges.max() >= len(embeddings):
        raise ValueError("edge_indices reference an unavailable node")
    normalized = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8
    )
    scores = np.sum(normalized[edges[:, 0]] * normalized[edges[:, 1]], axis=1)
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


def parent_choice_margins(cosine_scores: np.ndarray, edge_indices: np.ndarray) -> np.ndarray:
    """Score each candidate parent against the best alternative for its target."""
    scores = np.asarray(cosine_scores, dtype=np.float32).reshape(-1)
    edges = np.asarray(edge_indices, dtype=np.int64)
    if edges.shape != (len(scores), 2):
        raise ValueError("edge_indices and cosine_scores must describe the same edges")
    if not np.isfinite(scores).all():
        raise ValueError("cosine_scores must be finite")
    margins = np.zeros(len(scores), dtype=np.float32)
    by_target: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(edges[:, 1].tolist()):
        by_target[int(target)].append(index)
    for indices in by_target.values():
        if len(indices) == 1:
            continue
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), index))
        best, second = ordered[0], ordered[1]
        margins[best] = scores[best] - scores[second]
        for index in ordered[1:]:
            margins[index] = scores[index] - scores[best]
    return margins
