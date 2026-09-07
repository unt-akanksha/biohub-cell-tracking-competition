"""Deep temporal 3D detector with full-resolution heatmap and offset heads.

The design is intentionally independent of the public Biohub notebooks.  It
uses three consecutive isotropic frames, explicit temporal differences, a
deep ConvNeXt-style 3D pyramid, and a U-Net decoder.  Dense depth is placed at
lower resolutions so a 64-cubed volume remains practical on one A10G/T4.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_WIDTHS = (96, 192, 384, 768)
DEFAULT_DEPTHS = (3, 3, 9, 3)


class ChannelsFirstLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            values.movedim(1, -1),
            (values.shape[1],),
            self.weight,
            self.bias,
        ).movedim(-1, 1)


class ConvNeXtBlock3D(nn.Module):
    """Large-receptive-field residual block with inexpensive spatial mixing."""

    def __init__(self, channels: int, *, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, hidden)
        self.contract = nn.Linear(hidden, channels)
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values).movedim(1, -1)
        values = self.contract(F.gelu(self.expand(self.norm(values))))
        values = values * self.scale
        return residual + values.movedim(-1, 1)


class Stage(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        if channels <= 0 or depth <= 0:
            raise ValueError("stage channels and depth must be positive")
        self.blocks = nn.Sequential(
            *(ConvNeXtBlock3D(channels) for _ in range(depth))
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.blocks(values)


class Downsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ChannelsFirstLayerNorm(input_channels),
            nn.Conv3d(input_channels, output_channels, kernel_size=2, stride=2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.body(values)


class DecoderStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        *,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(
                output_channels + skip_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            ChannelsFirstLayerNorm(output_channels),
            nn.GELU(),
        )
        self.refine = Stage(output_channels, depth)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[-3:] != skip.shape[-3:]:
            raise ValueError("decoder and encoder spatial shapes differ")
        return self.refine(self.fuse(torch.cat((values, skip), dim=1)))


class TemporalPeakRankDetector(nn.Module):
    """Predict center-frame peaks and subvoxel offsets from a frame triplet."""

    def __init__(
        self,
        *,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        depths: Sequence[int] = DEFAULT_DEPTHS,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in widths)
        depths = tuple(int(value) for value in depths)
        if len(widths) != 4 or len(depths) != 4:
            raise ValueError("widths and depths must contain four values")
        if min(*widths, *depths) <= 0:
            raise ValueError("widths and depths must be positive")

        # prev, current, next, current-prev, next-current, and temporal range.
        self.stem = nn.Sequential(
            nn.Conv3d(6, widths[0], kernel_size=3, padding=1, bias=False),
            ChannelsFirstLayerNorm(widths[0]),
            nn.GELU(),
        )
        self.encoder = nn.ModuleList(
            Stage(widths[index], depths[index]) for index in range(4)
        )
        self.downsamples = nn.ModuleList(
            Downsample(widths[index], widths[index + 1]) for index in range(3)
        )
        self.decoder = nn.ModuleList(
            (
                DecoderStage(widths[3], widths[2], widths[2]),
                DecoderStage(widths[2], widths[1], widths[1]),
                DecoderStage(widths[1], widths[0], widths[0]),
            )
        )
        self.heatmap_head = nn.Conv3d(widths[0], 1, kernel_size=1)
        self.offset_head = nn.Conv3d(widths[0], 3, kernel_size=1)
        self.auxiliary_heads = nn.ModuleList(
            (nn.Conv3d(widths[2], 1, 1), nn.Conv3d(widths[1], 1, 1))
        )
        nn.init.normal_(self.heatmap_head.weight, std=1e-3)
        nn.init.constant_(self.heatmap_head.bias, -4.0)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        for head in self.auxiliary_heads:
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.constant_(head.bias, -4.0)

    @staticmethod
    def temporal_channels(frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1] != 3:
            raise ValueError("frames must have shape (B, 3, Z, Y, X)")
        previous, current, following = frames.unbind(dim=1)
        temporal_range = torch.maximum(
            torch.maximum(previous, current), following
        ) - torch.minimum(torch.minimum(previous, current), following)
        return torch.stack(
            (
                previous,
                current,
                following,
                current - previous,
                following - current,
                temporal_range,
            ),
            dim=1,
        )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        if any(int(size) % 8 for size in frames.shape[-3:]):
            raise ValueError("spatial dimensions must be divisible by eight")
        values = self.stem(self.temporal_channels(frames))
        skips: list[torch.Tensor] = []
        for index, stage in enumerate(self.encoder):
            values = stage(values)
            if index < 3:
                skips.append(values)
                values = self.downsamples[index](values)

        auxiliary: list[torch.Tensor] = []
        for index, decoder in enumerate(self.decoder):
            values = decoder(values, skips[-1 - index])
            if index < 2:
                auxiliary.append(self.auxiliary_heads[index](values))
        return {
            "logits": self.heatmap_head(values),
            "offsets": torch.tanh(self.offset_head(values)) * 0.5,
            "auxiliary_logits": tuple(auxiliary),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
