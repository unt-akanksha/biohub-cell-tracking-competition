"""Learned, count-preserving center enhancement for frozen 3D detections.

The implementation is paper-inspired but independent: it does not import
CELLECT code or weights.  Sparse organizer annotations supervise only matched
peak locations.  Unmatched detections are deliberately left unlabeled because
the training graphs are not exhaustive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


INPUT_VOXEL_UM = np.asarray((1.625, 1.625, 1.625), dtype=np.float32)
NATIVE_TO_INPUT = np.asarray((1.0, 0.25, 0.25), dtype=np.float32)


def _points(values: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite (N, 3) array")
    return result


def _patch_shape(values: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value < 3 or value % 2 == 0 for value in result):
        raise ValueError("patch shape must contain three odd values of at least three")
    return result


@dataclass(frozen=True)
class MatchedOffsets:
    peak_indices: np.ndarray
    annotation_indices: np.ndarray
    offsets_input: np.ndarray
    distances_um: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.peak_indices)
        if (
            self.peak_indices.shape != (count,)
            or self.annotation_indices.shape != (count,)
            or self.offsets_input.shape != (count, 3)
            or self.distances_um.shape != (count,)
        ):
            raise ValueError("matched-offset arrays have inconsistent shapes")


def match_annotated_peaks(
    peaks_input: np.ndarray | Sequence[Sequence[float]],
    annotations_native: np.ndarray | Sequence[Sequence[float]],
    *,
    maximum_distance_um: float = 5.0,
) -> MatchedOffsets:
    """Build one-to-one offset labels without treating unmatched peaks as negatives."""

    peaks = _points(peaks_input, "peaks_input")
    annotations = _points(annotations_native, "annotations_native")
    if maximum_distance_um <= 0:
        raise ValueError("maximum_distance_um must be positive")
    if not len(peaks) or not len(annotations):
        return MatchedOffsets(
            peak_indices=np.empty(0, dtype=np.int64),
            annotation_indices=np.empty(0, dtype=np.int64),
            offsets_input=np.empty((0, 3), dtype=np.float32),
            distances_um=np.empty(0, dtype=np.float32),
        )

    annotations_input = annotations * NATIVE_TO_INPUT
    distances = np.linalg.norm(
        (peaks[:, None, :] - annotations_input[None, :, :]) * INPUT_VOXEL_UM,
        axis=-1,
    )
    peak_indices, annotation_indices = linear_sum_assignment(distances)
    accepted = distances[peak_indices, annotation_indices] <= maximum_distance_um
    peak_indices = peak_indices[accepted].astype(np.int64, copy=False)
    annotation_indices = annotation_indices[accepted].astype(np.int64, copy=False)
    offsets = annotations_input[annotation_indices] - peaks[peak_indices]
    return MatchedOffsets(
        peak_indices=peak_indices,
        annotation_indices=annotation_indices,
        offsets_input=offsets.astype(np.float32, copy=False),
        distances_um=distances[peak_indices, annotation_indices].astype(np.float32, copy=False),
    )


def extract_center_patches(
    intensity: np.ndarray,
    probability: np.ndarray,
    centers: np.ndarray | Sequence[Sequence[float]],
    *,
    patch_shape: Sequence[int] = (7, 7, 7),
) -> np.ndarray:
    """Extract locally normalized raw/probability patches around rounded peaks."""

    raw = np.asarray(intensity, dtype=np.float32)
    heatmap = np.asarray(probability, dtype=np.float32)
    points = _points(centers, "centers")
    shape = _patch_shape(patch_shape)
    if raw.ndim != 3 or heatmap.shape != raw.shape:
        raise ValueError("intensity and probability must be matching 3D volumes")
    if not np.isfinite(raw).all() or not np.isfinite(heatmap).all():
        raise ValueError("input volumes must be finite")
    if not len(points):
        return np.empty((0, 2, *shape), dtype=np.float32)

    radius = np.asarray(shape, dtype=np.int64) // 2
    padded_raw = np.pad(raw, tuple((int(value), int(value)) for value in radius), mode="reflect")
    padded_probability = np.pad(
        heatmap,
        tuple((int(value), int(value)) for value in radius),
        mode="reflect",
    )
    result = np.empty((len(points), 2, *shape), dtype=np.float32)
    integer_centers = np.rint(points).astype(np.int64)
    if np.any(integer_centers < 0) or np.any(integer_centers >= np.asarray(raw.shape)):
        raise ValueError("center lies outside the input volume")
    for index, center in enumerate(integer_centers):
        slices = tuple(
            slice(int(center[axis]), int(center[axis] + shape[axis]))
            for axis in range(3)
        )
        raw_patch = padded_raw[slices]
        probability_patch = padded_probability[slices]
        raw_low, raw_high = np.quantile(raw_patch, (0.1, 0.99))
        raw_patch = np.clip(
            (raw_patch - raw_low) / max(float(raw_high - raw_low), 1e-6),
            0.0,
            1.0,
        )
        probability_high = max(float(probability_patch.max()), 1e-6)
        result[index, 0] = raw_patch
        result[index, 1] = np.clip(probability_patch / probability_high, 0.0, 1.0)
    return result


def offset_target_distribution(
    offsets: "object",
    *,
    patch_shape: Sequence[int] = (7, 7, 7),
    sigma_voxels: float = 0.65,
):
    """Create normalized sub-voxel Gaussian targets for center enhancement."""

    import torch

    shape = _patch_shape(patch_shape)
    target_offsets = torch.as_tensor(offsets)
    if target_offsets.ndim != 2 or target_offsets.shape[1] != 3:
        raise ValueError("offsets must have shape (N, 3)")
    if sigma_voxels <= 0:
        raise ValueError("sigma_voxels must be positive")
    radii = [(value - 1) / 2.0 for value in shape]
    if target_offsets.numel() and torch.any(
        torch.abs(target_offsets)
        > target_offsets.new_tensor(radii)[None, :]
    ):
        raise ValueError("target offset lies outside the enhancement patch")
    axes = [
        torch.arange(value, device=target_offsets.device, dtype=target_offsets.dtype)
        - radius
        for value, radius in zip(shape, radii)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    squared = torch.sum(
        (grid[None, ...] - target_offsets[:, None, None, None, :]) ** 2,
        dim=-1,
    )
    targets = torch.exp(-0.5 * squared / sigma_voxels**2)
    return targets / targets.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)


def spatial_expectation(logits: "object"):
    """Convert a 3D center logit volume to a bounded sub-voxel offset."""

    import torch

    if logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError("logits must have shape (N, 1, Z, Y, X)")
    shape = tuple(int(value) for value in logits.shape[2:])
    _patch_shape(shape)
    probabilities = torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits)
    axes = [
        torch.arange(value, device=logits.device, dtype=logits.dtype)
        - (value - 1) / 2.0
        for value in shape
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)[None]
    return torch.sum(probabilities * grid, dim=(2, 3, 4))


def build_center_enhancement_model(*, channels: int = 32):
    """Construct a compact residual 3D center-enhancement network."""

    import torch
    from torch import nn

    if channels < 8 or channels % 8:
        raise ValueError("channels must be a multiple of eight and at least eight")

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv3d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
            )

        def forward(self, values):
            return torch.nn.functional.silu(values + self.body(values))

    class CenterEnhancementModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv3d(2, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                ResidualBlock(),
                ResidualBlock(),
                nn.Conv3d(channels, 1, 1),
            )

        def forward(self, patches):
            if patches.ndim != 5 or patches.shape[1] != 2:
                raise ValueError("patches must have shape (N, 2, Z, Y, X)")
            logits = self.network(patches)
            return {"logits": logits, "offsets": spatial_expectation(logits)}

    return CenterEnhancementModel()


def center_enhancement_loss(
    logits: "object",
    target_offsets: "object",
    *,
    sigma_voxels: float = 0.65,
    offset_weight: float = 0.5,
):
    """Dense center cross-entropy plus robust sub-voxel offset regression."""

    import torch

    if offset_weight < 0:
        raise ValueError("offset_weight cannot be negative")
    target_offsets = torch.as_tensor(target_offsets, device=logits.device, dtype=logits.dtype)
    targets = offset_target_distribution(
        target_offsets,
        patch_shape=logits.shape[2:],
        sigma_voxels=sigma_voxels,
    )
    log_probabilities = torch.log_softmax(logits[:, 0].flatten(1), dim=-1)
    dense = -(targets.flatten(1) * log_probabilities).sum(dim=1).mean()
    predicted_offsets = spatial_expectation(logits)
    regression = torch.nn.functional.smooth_l1_loss(predicted_offsets, target_offsets)
    total = dense + offset_weight * regression
    return {
        "loss": total,
        "dense_loss": dense.detach(),
        "offset_loss": regression.detach(),
        "predicted_offsets": predicted_offsets,
    }


def apply_bounded_offsets(
    centers: np.ndarray | Sequence[Sequence[float]],
    offsets: np.ndarray | Sequence[Sequence[float]],
    *,
    volume_shape: Sequence[int],
    maximum_offset_voxels: float = 2.0,
) -> np.ndarray:
    """Apply learned offsets while preserving identities and staying in bounds."""

    points = _points(centers, "centers")
    delta = _points(offsets, "offsets")
    shape = np.asarray(tuple(int(value) for value in volume_shape), dtype=np.float32)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("volume_shape must contain three positive values")
    if delta.shape != points.shape:
        raise ValueError("centers and offsets must have the same shape")
    if maximum_offset_voxels <= 0:
        raise ValueError("maximum_offset_voxels must be positive")
    bounded = np.clip(delta, -maximum_offset_voxels, maximum_offset_voxels)
    return np.clip(points + bounded, 0.0, shape[None, :] - 1.0).astype(np.float32)
