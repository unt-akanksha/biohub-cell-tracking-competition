"""Blob-aware extension of the independent temporal 3D peak detector."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:
    from model import (
        DEFAULT_DEPTHS,
        DEFAULT_WIDTHS,
        ChannelsFirstLayerNorm,
        TemporalPeakRankDetector,
    )
except ModuleNotFoundError:
    from research.peak_rank_detection.model import (
        DEFAULT_DEPTHS,
        DEFAULT_WIDTHS,
        ChannelsFirstLayerNorm,
        TemporalPeakRankDetector,
    )


MODEL_FAMILY = "blob_aware_temporal_peak_rank_v11"


def replicated_average(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Average a `(B, Z, Y, X)` field without zero-padding border artifacts."""

    if values.ndim != 4 or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("values must be BZYX and kernel_size must be positive odd")
    padding = kernel_size // 2
    padded = F.pad(
        values[:, None],
        (padding, padding, padding, padding, padding, padding),
        mode="replicate",
    )
    return F.avg_pool3d(padded, kernel_size=kernel_size, stride=1)[:, 0]


def blob_bandpass(values: torch.Tensor) -> torch.Tensor:
    """Fixed local contrast matching the strongest clean real-crop diagnostic."""

    return replicated_average(values, 3) - replicated_average(values, 9)


class BlobAwareTemporalPeakRankDetector(TemporalPeakRankDetector):
    """Add current-frame and temporal-mean blob priors to the learned stem."""

    def __init__(
        self,
        *,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        depths: Sequence[int] = DEFAULT_DEPTHS,
    ) -> None:
        super().__init__(widths=widths, depths=depths)
        widths = tuple(int(value) for value in widths)
        self.stem = nn.Sequential(
            nn.Conv3d(8, widths[0], kernel_size=3, padding=1, bias=False),
            ChannelsFirstLayerNorm(widths[0]),
            nn.GELU(),
        )

    @staticmethod
    def temporal_channels(frames: torch.Tensor) -> torch.Tensor:
        learned = TemporalPeakRankDetector.temporal_channels(frames)
        current = frames[:, 1]
        temporal_mean = frames.mean(dim=1)
        return torch.cat(
            (
                learned,
                blob_bandpass(current)[:, None],
                blob_bandpass(temporal_mean)[:, None],
            ),
            dim=1,
        )
