"""Detection-only reconstruction of the public TemporalUNet3D teacher.

The public checkpoint also contains the association transformer.  Positive-
unlabeled detector adaptation needs only the frozen U-Net and detection head,
so this module loads exactly those state entries and rejects any missing or
shape-mismatched detector tensor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels = x.shape[:3]
        spatial = x.shape[3:]
        locations = math.prod(spatial)
        hidden = (
            x.reshape(batch, frames, channels, locations)
            .permute(0, 3, 1, 2)
            .reshape(batch * locations, frames, channels)
        )
        hidden = self.norm(hidden)
        hidden, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = (
            hidden.reshape(batch, locations, frames, channels)
            .permute(0, 2, 3, 1)
            .reshape(batch, frames, channels, *spatial)
        )
        return x + hidden


class TemporalUNet3D(nn.Module):
    """Exact detection-backbone topology from the audited public source."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128),
        *,
        gradient_checkpointing: bool = False,
        skip_fullres_temporal: bool = True,
    ) -> None:
        super().__init__()
        widths = list(layers)
        if len(widths) < 2:
            raise ValueError("layers must contain at least two stages")
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        previous = in_channels
        for index, channels in enumerate(widths):
            self.encoder_blocks.append(_conv_block(previous, channels))
            self.temporal_blocks.append(
                nn.Identity()
                if skip_fullres_temporal and index == 0
                else TemporalAttention(channels)
            )
            previous = channels
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for index in range(len(widths) - 1, 0, -1):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            )
            self.decoder_blocks.append(
                _conv_block(widths[index] + widths[index - 1], widths[index - 1])
            )
        self.head = nn.Conv3d(widths[0], out_channels, kernel_size=1)

    def _run(self, block: nn.Module, values: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return gradient_checkpoint(block, values, use_reentrant=False)
        return block(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, frames = values.shape[:2]
        values = values.reshape(batch * frames, *values.shape[2:])
        skips: list[torch.Tensor] = []
        for index, (block, temporal) in enumerate(
            zip(self.encoder_blocks, self.temporal_blocks)
        ):
            if index:
                values = self.pool(values)
            values = self._run(block, values)
            values = temporal(values.reshape(batch, frames, *values.shape[1:])).reshape(
                batch * frames, *values.shape[1:]
            )
            if index < len(self.encoder_blocks) - 1:
                skips.append(values)
        for upsample, block, skip in zip(
            self.upsamples, self.decoder_blocks, reversed(skips)
        ):
            values = upsample(values)
            if values.shape[2:] != skip.shape[2:]:
                values = F.interpolate(
                    values, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            values = self._run(block, torch.cat([values, skip], dim=1))
        values = self.head(values)
        return values.reshape(batch, frames, *values.shape[1:])


class PublicTeacherDetector(nn.Module):
    def __init__(
        self, *, out_channels: int = 32, layers: Sequence[int] = (32, 64, 128)
    ) -> None:
        super().__init__()
        self.unet = TemporalUNet3D(out_channels=out_channels, layers=layers)
        self.detect_head = nn.Conv3d(out_channels, 1, kernel_size=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return per-frame detection logits for ``(B,T,Z,Y,X)`` input."""

        if frames.ndim != 5:
            raise ValueError("teacher input must have shape (B, T, Z, Y, X)")
        features = self.unet(frames.unsqueeze(2))
        batch, time = features.shape[:2]
        logits = self.detect_head(features.reshape(batch * time, *features.shape[2:]))
        return logits.reshape(batch, time, *logits.shape[2:])


def load_public_teacher(
    weights_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    out_channels: int = 32,
    layers: Sequence[int] = (32, 64, 128),
) -> PublicTeacherDetector:
    """Load and freeze the detector subset of an audited public checkpoint."""

    state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("teacher checkpoint must contain a state mapping")
    detector_state = {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(("unet.", "detect_head."))
    }
    if not detector_state:
        raise ValueError("checkpoint contains no detector tensors")
    model = PublicTeacherDetector(out_channels=out_channels, layers=layers)
    model.load_state_dict(detector_state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


@torch.no_grad()
def teacher_probabilities(
    model: PublicTeacherDetector,
    normalized_frames: torch.Tensor,
    *,
    yx_tta: bool = True,
) -> torch.Tensor:
    """Match public inference: average Y/X-flip logits, then apply sigmoid."""

    logits = model(normalized_frames)
    if yx_tta:
        for dims in ((-1,), (-2,), (-2, -1)):
            logits = logits + model(normalized_frames.flip(dims)).flip(dims)
        logits = logits / 4.0
    return torch.sigmoid(logits)
