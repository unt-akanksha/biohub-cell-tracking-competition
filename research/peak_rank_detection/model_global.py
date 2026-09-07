"""Blob-aware temporal detector with learned global bottleneck context.

The convolutional pyramid retains precise local geometry while two compact
attention blocks exchange evidence across the 8-cubed bottleneck field.  This
keeps full-resolution memory bounded for 64-cubed training crops and gives
faint or crowded cells access to whole-crop context before decoding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

try:
    from model import DEFAULT_DEPTHS, DEFAULT_WIDTHS
except ModuleNotFoundError:
    from research.peak_rank_detection.model import DEFAULT_DEPTHS, DEFAULT_WIDTHS

try:
    from model_blob import BlobAwareTemporalPeakRankDetector
except ModuleNotFoundError:
    from research.peak_rank_detection.model_blob import (
        BlobAwareTemporalPeakRankDetector,
    )


MODEL_FAMILY = "blob_global_context_temporal_peak_rank_v13"


class GlobalContextBlock3D(nn.Module):
    """Pre-norm global attention with conservative LayerScale residuals."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 16,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads:
            raise ValueError("channels must be positive and divisible by heads")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        hidden = channels * expansion
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_scale = nn.Parameter(torch.full((channels,), 1e-4))
        self.mlp_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.mlp_scale = nn.Parameter(torch.full((channels,), 1e-4))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5 or values.shape[1] != self.attention.embed_dim:
            raise ValueError("values must have shape (B, C, Z, Y, X)")
        shape = values.shape
        tokens = values.flatten(2).transpose(1, 2)
        normalized = self.attention_norm(tokens)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        tokens = tokens + attended * self.attention_scale
        tokens = tokens + self.mlp(self.mlp_norm(tokens)) * self.mlp_scale
        return tokens.transpose(1, 2).reshape(shape)


class BlobGlobalTemporalPeakRankDetector(BlobAwareTemporalPeakRankDetector):
    """Add global low-resolution context to the blob-aware temporal U-Net."""

    def __init__(
        self,
        *,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        depths: Sequence[int] = DEFAULT_DEPTHS,
        global_blocks: int = 2,
    ) -> None:
        super().__init__(widths=widths, depths=depths)
        widths = tuple(int(value) for value in widths)
        if global_blocks <= 0:
            raise ValueError("global_blocks must be positive")
        heads = 16 if widths[-1] % 16 == 0 else 8
        if widths[-1] % heads:
            raise ValueError("bottleneck width is incompatible with attention heads")
        local_stage = self.encoder[-1]
        self.encoder[-1] = nn.Sequential(
            local_stage,
            *(
                GlobalContextBlock3D(widths[-1], heads=heads, expansion=2)
                for _ in range(global_blocks)
            ),
        )
