"""A high-resolution 3D detector built around microscopy-pretrained SpatialDINO.

The public Biohub detector is used only to construct conservative training
targets elsewhere.  This model has an independent architecture and learned
weights: a SpatialDINO ViT-S/8 semantic path, a raw-image convolutional pyramid,
and a UNETR-like decoder that predicts an isotropic full-resolution heatmap.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

try:
    from encoder import SpatialDinoViTS8
except ModuleNotFoundError:
    from research.spatialdino_association.encoder import SpatialDinoViTS8


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(values) + self.projection(values))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.refine = ResidualConv3d(out_channels, out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(values))


class UpFuseBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        self.refine = ResidualConv3d(out_channels + skip_channels, out_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[-3:] != skip.shape[-3:]:
            raise ValueError("decoder and skip feature shapes do not match")
        return self.refine(torch.cat((values, skip), dim=1))


class TokenProjection(nn.Module):
    """Project and resize a transformer grid for convolutional fusion."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.scale_factor = int(scale_factor)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.projection(values)
        if self.scale_factor != 1:
            values = F.interpolate(
                values,
                scale_factor=self.scale_factor,
                mode="trilinear",
                align_corners=False,
            )
        return values


class HybridSpatialDinoDetector(nn.Module):
    """SpatialDINO ViT-S/8 plus raw 3D pyramid and full-resolution decoder."""

    intermediate_blocks = (2, 5, 8, 11)

    def __init__(
        self,
        encoder: SpatialDinoViTS8,
        *,
        widths: tuple[int, int, int, int] = (24, 48, 96, 192),
    ) -> None:
        super().__init__()
        if len(widths) != 4 or any(width <= 0 for width in widths):
            raise ValueError("widths must contain four positive channel counts")
        self.encoder = encoder
        c0, c1, c2, c3 = map(int, widths)
        embed_dim = int(encoder.embed_dim)

        self.raw_stem = ResidualConv3d(1, c0)
        self.raw_down1 = DownBlock(c0, c1)
        self.raw_down2 = DownBlock(c1, c2)
        self.raw_down3 = DownBlock(c2, c3)

        self.token_deep = TokenProjection(embed_dim, c3, 1)
        self.token_16 = TokenProjection(embed_dim, c2, 2)
        self.token_32 = TokenProjection(embed_dim, c1, 4)
        self.token_64 = TokenProjection(embed_dim, c0, 8)

        self.bottleneck = ResidualConv3d(c3 * 2, c3)
        self.decode_16 = UpFuseBlock(c3, c2 * 2, c2)
        self.decode_32 = UpFuseBlock(c2, c1 * 2, c1)
        self.decode_64 = UpFuseBlock(c1, c0 * 2, c0)
        self.heatmap_head = nn.Conv3d(c0, 1, kernel_size=1)
        # A sparse detector should not begin at sigmoid(0)=0.5 everywhere.
        # The small head weights preserve the learned feature scale while the
        # low foreground prior prevents a randomly decoded peak field.
        nn.init.normal_(self.heatmap_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.heatmap_head.bias, -4.0)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError("volume must have shape (B, 1, Z, Y, X)")
        if any(int(size) % 8 for size in volume.shape[-3:]):
            raise ValueError("volume dimensions must be divisible by 8")

        raw_64 = self.raw_stem(volume)
        raw_32 = self.raw_down1(raw_64)
        raw_16 = self.raw_down2(raw_32)
        raw_8 = self.raw_down3(raw_16)
        token_early, token_mid, token_late, token_final = self.encoder.forward_intermediates(
            volume, block_indices=self.intermediate_blocks
        )

        values = self.bottleneck(torch.cat((raw_8, self.token_deep(token_final)), dim=1))
        values = self.decode_16(
            values, torch.cat((raw_16, self.token_16(token_late)), dim=1)
        )
        values = self.decode_32(
            values, torch.cat((raw_32, self.token_32(token_mid)), dim=1)
        )
        values = self.decode_64(
            values, torch.cat((raw_64, self.token_64(token_early)), dim=1)
        )
        return self.heatmap_head(values)


def set_detector_training_phase(
    model: HybridSpatialDinoDetector,
    *,
    unfreeze_last_encoder_blocks: int = 0,
) -> dict[str, int]:
    """Freeze SpatialDINO by default and optionally tune only its last blocks."""

    if not 0 <= unfreeze_last_encoder_blocks <= model.encoder.depth:
        raise ValueError("unfreeze_last_encoder_blocks is outside the encoder depth")
    model.requires_grad_(True)
    model.encoder.requires_grad_(False)
    if unfreeze_last_encoder_blocks:
        for block in model.encoder.blocks[-unfreeze_last_encoder_blocks:]:
            block.requires_grad_(True)
        model.encoder.norm.requires_grad_(True)

    encoder_trainable = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )
    decoder_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    )
    return {
        "encoder_trainable_parameters": encoder_trainable,
        "decoder_trainable_parameters": decoder_trainable,
        "total_trainable_parameters": encoder_trainable + decoder_trainable,
    }


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
