"""Independent physical-scale 3D patch encoder for Biohub association.

The model is trained on image crops centered at graph nodes; it does not ingest
public prediction identities or code. Physical resampling gives synthetic
isotropic volumes and real anisotropic volumes the same field of view.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _positive_tuple3(values: Sequence[int | float], name: str) -> tuple:
    result = tuple(values)
    if len(result) != 3 or any(float(value) <= 0 for value in result):
        raise ValueError(f"{name} must contain three positive values")
    return result


def sample_physical_patches(
    volume: np.ndarray | torch.Tensor,
    centers_zyx_voxel: np.ndarray | torch.Tensor,
    *,
    voxel_size_zyx_um: Sequence[float],
    output_shape_zyx: Sequence[int] = (17, 17, 17),
    half_extent_zyx_um: Sequence[float] = (8.0, 8.0, 8.0),
    chunk_size: int = 64,
) -> torch.Tensor:
    """Resample center-aligned channel volumes onto one physical grid.

    A single volume may be ``(Z, Y, X)``. Temporal or multimodal input may be
    ``(C, Z, Y, X)``; every channel uses the same node-centered sampling grid
    and is normalized independently so intensity drift cannot dominate motion.
    """

    image = torch.as_tensor(volume, dtype=torch.float32)
    centers = torch.as_tensor(
        centers_zyx_voxel, dtype=torch.float32, device=image.device
    )
    voxel_size = _positive_tuple3(voxel_size_zyx_um, "voxel_size_zyx_um")
    output_shape = tuple(
        int(value) for value in _positive_tuple3(output_shape_zyx, "output_shape_zyx")
    )
    half_extent = _positive_tuple3(half_extent_zyx_um, "half_extent_zyx_um")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("volume must have shape (Z, Y, X) or (C, Z, Y, X)")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_zyx_voxel must have shape (N, 3)")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not torch.isfinite(image).all() or not torch.isfinite(centers).all():
        raise ValueError("volume and centers must be finite")
    if len(centers) == 0:
        return image.new_empty((0, image.shape[0], *output_shape))

    axis_offsets = [
        torch.linspace(-float(extent), float(extent), count, device=image.device)
        / float(spacing)
        for extent, count, spacing in zip(
            half_extent, output_shape, voxel_size, strict=True
        )
    ]
    offset_grid = torch.stack(
        torch.meshgrid(*axis_offsets, indexing="ij"), dim=-1
    )
    spatial_shape = tuple(int(value) for value in image.shape[-3:])
    input_tensor = image[None]
    patches: list[torch.Tensor] = []
    for start in range(0, len(centers), chunk_size):
        batch_centers = centers[start : start + chunk_size]
        coordinates = batch_centers[:, None, None, None, :] + offset_grid[None]
        normalized_axes: list[torch.Tensor] = []
        for axis, size in enumerate(spatial_shape):
            if size == 1:
                normalized_axes.append(torch.zeros_like(coordinates[..., axis]))
            else:
                normalized_axes.append(
                    2.0 * coordinates[..., axis] / float(size - 1) - 1.0
                )
        grid = torch.stack(
            (normalized_axes[2], normalized_axes[1], normalized_axes[0]), dim=-1
        )
        sampled = F.grid_sample(
            input_tensor.expand(len(batch_centers), -1, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        patches.append(sampled)
    result = torch.cat(patches, dim=0)
    means = result.mean(dim=(2, 3, 4), keepdim=True)
    scales = result.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)
    return ((result - means) / scales).clamp(-6.0, 6.0)


def temporal_context_volume(
    volumes: np.ndarray | torch.Tensor,
    timepoint: int,
    *,
    offsets: Sequence[int] = (-1, 0, 1),
) -> np.ndarray | torch.Tensor:
    """Stack boundary-clamped movie frames as channels around one timepoint."""

    if len(offsets) == 0 or 0 not in offsets:
        raise ValueError("temporal offsets must be nonempty and include zero")
    if len(getattr(volumes, "shape", ())) != 4:
        raise ValueError("movie volumes must have shape (T, Z, Y, X)")
    frame_count = int(volumes.shape[0])
    if frame_count <= 0 or not 0 <= int(timepoint) < frame_count:
        raise ValueError("timepoint is outside the movie")
    indices = [min(max(int(timepoint) + int(offset), 0), frame_count - 1) for offset in offsets]
    if isinstance(volumes, torch.Tensor):
        index = torch.as_tensor(indices, dtype=torch.long, device=volumes.device)
        return volumes.index_select(0, index)
    return np.stack([np.asarray(volumes[index]) for index in indices], axis=0)


def physical_candidate_masks(
    source_coords_zyx_voxel: np.ndarray,
    target_coords_zyx_voxel: np.ndarray,
    positive_edges: np.ndarray,
    *,
    voxel_size_zyx_um: Sequence[float],
    radius_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build geometric candidates while proving every labeled edge is retained.

    ``positive_edges`` contains source-row/target-row pairs for one frame
    transition, not global node identifiers.
    """

    source = np.asarray(source_coords_zyx_voxel, dtype=np.float32)
    target = np.asarray(target_coords_zyx_voxel, dtype=np.float32)
    positives = np.asarray(positive_edges, dtype=np.int64).reshape(-1, 2)
    scale = np.asarray(
        _positive_tuple3(voxel_size_zyx_um, "voxel_size_zyx_um"),
        dtype=np.float32,
    )
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source coordinates must have shape (N, 3)")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target coordinates must have shape (M, 3)")
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("coordinates must be finite")
    if positives.size and (
        positives.min() < 0
        or positives[:, 0].max() >= len(source)
        or positives[:, 1].max() >= len(target)
    ):
        raise ValueError("positive edge references an unavailable row")
    distances = np.linalg.norm(
        source[:, None, :] * scale[None, None, :]
        - target[None, :, :] * scale[None, None, :],
        axis=2,
    )
    candidates = distances <= float(radius_um)
    positive_mask = np.zeros_like(candidates, dtype=bool)
    if len(positives):
        positive_mask[positives[:, 0], positives[:, 1]] = True
    if np.any(positive_mask & ~candidates):
        missed = np.argwhere(positive_mask & ~candidates)
        raise ValueError(
            f"candidate radius omitted {len(missed)} ground-truth links; "
            f"largest positive distance={float(distances[positive_mask].max()):.6f} um"
        )
    return candidates, positive_mask


class ResidualBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1):
        super().__init__()
        groups = next(
            value for value in (8, 4, 2, 1) if output_channels % value == 0
        )
        self.main = nn.Sequential(
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(inputs) + self.skip(inputs), inplace=True)


class PhysicalPatchAssociationModel(nn.Module):
    """A 3D residual encoder with normalized link and division representations."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        base_channels: int = 64,
        embedding_channels: int = 256,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels <= 0 or embedding_channels <= 0:
            raise ValueError("channel counts must be positive")
        self.input_channels = int(input_channels)
        groups = next(value for value in (8, 4, 2, 1) if base_channels % value == 0)
        self.stem = nn.Sequential(
            nn.Conv3d(
                self.input_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, base_channels),
            nn.SiLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            ResidualBlock3D(base_channels, base_channels),
            ResidualBlock3D(base_channels, base_channels * 2, stride=2),
            ResidualBlock3D(base_channels * 2, base_channels * 2),
            ResidualBlock3D(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock3D(base_channels * 4, base_channels * 4),
            ResidualBlock3D(base_channels * 4, base_channels * 8, stride=2),
        )
        final_channels = base_channels * 8
        self.projection = nn.Sequential(
            nn.Linear(final_channels, final_channels),
            nn.SiLU(inplace=True),
            nn.Linear(final_channels, embedding_channels),
        )
        self.division = nn.Sequential(
            nn.Linear(final_channels, base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels * 2, 1),
        )
        nn.init.constant_(self.division[-1].bias, -4.0)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.ndim != 5 or patches.shape[1] != self.input_channels:
            raise ValueError(
                f"patches must have shape (N, {self.input_channels}, Z, Y, X)"
            )
        hidden = self.encoder(self.stem(patches)).mean(dim=(2, 3, 4))
        embeddings = F.normalize(self.projection(hidden), p=2, dim=1, eps=1e-8)
        return embeddings, self.division(hidden).squeeze(1)

    def pair_logits(
        self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
            raise ValueError("embeddings must be matrices")
        if source_embeddings.shape[1] != target_embeddings.shape[1]:
            raise ValueError("embedding widths must match")
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * (
            F.normalize(source_embeddings, dim=1, eps=1e-8)
            @ F.normalize(target_embeddings, dim=1, eps=1e-8).transpose(0, 1)
        )
