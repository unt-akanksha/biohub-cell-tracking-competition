"""Multi-scale blob-aware global detector for real-domain cell variation."""

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

try:
    from model_global import BlobGlobalTemporalPeakRankDetector
except ModuleNotFoundError:
    from research.peak_rank_detection.model_global import (
        BlobGlobalTemporalPeakRankDetector,
    )


MODEL_FAMILY = "multiscale_blob_global_context_temporal_peak_rank_v15"
MULTISCALE_BANDS = ((3, 9), (5, 13), (7, 15))


def separable_replicated_average(
    values: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    """Exact separable box average for a ``(B,Z,Y,X)`` field."""

    if values.ndim != 4 or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("values must be BZYX and kernel_size must be positive odd")
    padding = kernel_size // 2
    pooled = values[:, None]
    for pad, kernel in (
        ((0, 0, 0, 0, padding, padding), (kernel_size, 1, 1)),
        ((0, 0, padding, padding, 0, 0), (1, kernel_size, 1)),
        ((padding, padding, 0, 0, 0, 0), (1, 1, kernel_size)),
    ):
        pooled = F.avg_pool3d(F.pad(pooled, pad, mode="replicate"), kernel, stride=1)
    return pooled[:, 0]


def multiscale_blob_channels(frames: torch.Tensor) -> torch.Tensor:
    """Return current/temporal-mean bandpasses at three fixed physical scales."""

    if frames.ndim != 5 or frames.shape[1] != 3:
        raise ValueError("frames must have shape (B,3,Z,Y,X)")
    batch = frames.shape[0]
    fields = torch.cat((frames[:, 1], frames.mean(dim=1)), dim=0)
    averages = {
        kernel: separable_replicated_average(fields, kernel)
        for kernel in sorted({value for band in MULTISCALE_BANDS for value in band})
    }
    channels: list[torch.Tensor] = []
    for inner, outer in MULTISCALE_BANDS:
        response = averages[inner] - averages[outer]
        channels.extend((response[:batch], response[batch:]))
    return torch.stack(channels, dim=1)


class MultiscaleBlobGlobalTemporalPeakRankDetector(
    BlobGlobalTemporalPeakRankDetector
):
    """Global ConvNeXt U-Net with efficient fixed multi-scale blob evidence."""

    def __init__(
        self,
        *,
        widths: Sequence[int] = DEFAULT_WIDTHS,
        depths: Sequence[int] = DEFAULT_DEPTHS,
        global_blocks: int = 2,
    ) -> None:
        super().__init__(
            widths=widths,
            depths=depths,
            global_blocks=global_blocks,
        )
        widths = tuple(int(value) for value in widths)
        self.stem = nn.Sequential(
            nn.Conv3d(12, widths[0], kernel_size=3, padding=1, bias=False),
            ChannelsFirstLayerNorm(widths[0]),
            nn.GELU(),
        )

    @staticmethod
    def temporal_channels(frames: torch.Tensor) -> torch.Tensor:
        learned = TemporalPeakRankDetector.temporal_channels(frames)
        return torch.cat((learned, multiscale_blob_channels(frames)), dim=1)
