"""Safe-rank detector with temporal-minimum local-SNR evidence."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

try:
    from model import DEFAULT_DEPTHS, DEFAULT_WIDTHS, ChannelsFirstLayerNorm
    from model_multiscale import MULTISCALE_BANDS, separable_replicated_average
    from model_safe_rank import SafeRankMultiscaleBlobGlobalDetector
except ModuleNotFoundError:
    from research.peak_rank_detection.model import (
        DEFAULT_DEPTHS,
        DEFAULT_WIDTHS,
        ChannelsFirstLayerNorm,
    )
    from research.peak_rank_detection.model_multiscale import (
        MULTISCALE_BANDS,
        separable_replicated_average,
    )
    from research.peak_rank_detection.model_safe_rank import (
        SafeRankMultiscaleBlobGlobalDetector,
    )


MODEL_FAMILY = "temporal_min_local_snr_safe_rank_multiscale_global_peak_rank_v23"


def temporal_minimum_local_snr_channels(frames: torch.Tensor) -> torch.Tensor:
    """Return three locally normalized bands stable across the frame triplet."""

    if frames.ndim != 5 or frames.shape[1] != 3:
        raise ValueError("frames must have shape (B,3,Z,Y,X)")
    stable = frames.amin(dim=1)
    averages = {
        kernel: separable_replicated_average(stable, kernel)
        for kernel in sorted({value for band in MULTISCALE_BANDS for value in band})
    }
    square_averages = {
        outer: separable_replicated_average(stable.square(), outer)
        for _, outer in MULTISCALE_BANDS
    }
    channels = []
    for inner, outer in MULTISCALE_BANDS:
        local_variance = (
            square_averages[outer] - averages[outer].square()
        ).clamp_min(0.0)
        local_scale = local_variance.sqrt()
        floors = []
        for sample_scale in local_scale:
            positive = sample_scale[sample_scale > 0.0]
            floor = (
                positive.float().quantile(0.25)
                if len(positive)
                else sample_scale.new_tensor(1e-3)
            )
            floors.append(floor.clamp_min(1e-3))
        floor_tensor = torch.stack(floors).to(local_scale.dtype)[:, None, None, None]
        channels.append(
            (averages[inner] - averages[outer])
            / torch.maximum(local_scale, floor_tensor)
        )
    return torch.stack(channels, dim=1)


class TemporalMinimumLocalSnrSafeRankDetector(SafeRankMultiscaleBlobGlobalDetector):
    """Expose stable local-SNR cell evidence as three extra fixed channels."""

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
            nn.Conv3d(15, widths[0], kernel_size=3, padding=1, bias=False),
            ChannelsFirstLayerNorm(widths[0]),
            nn.GELU(),
        )

    def temporal_channels(self, frames: torch.Tensor) -> torch.Tensor:
        existing = super().temporal_channels(frames)
        stable = temporal_minimum_local_snr_channels(frames)
        return torch.cat((existing, stable), dim=1)
