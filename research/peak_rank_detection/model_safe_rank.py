"""Multiscale global detector exposing fixed evidence for safe PU ranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

try:
    from model import DEFAULT_DEPTHS, DEFAULT_WIDTHS, TemporalPeakRankDetector
except ModuleNotFoundError:
    from research.peak_rank_detection.model import (
        DEFAULT_DEPTHS,
        DEFAULT_WIDTHS,
        TemporalPeakRankDetector,
    )

try:
    from model_multiscale import (
        MultiscaleBlobGlobalTemporalPeakRankDetector,
        multiscale_blob_channels,
    )
except ModuleNotFoundError:
    from research.peak_rank_detection.model_multiscale import (
        MultiscaleBlobGlobalTemporalPeakRankDetector,
        multiscale_blob_channels,
    )


MODEL_FAMILY = "safe_rank_multiscale_blob_global_temporal_peak_rank_v17"


class SafeRankMultiscaleBlobGlobalDetector(
    MultiscaleBlobGlobalTemporalPeakRankDetector
):
    """Expose the fixed avg5-avg13 response used to identify dark negatives."""

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
        self._safe_negative_evidence: torch.Tensor | None = None

    def temporal_channels(self, frames: torch.Tensor) -> torch.Tensor:
        learned = TemporalPeakRankDetector.temporal_channels(frames)
        multiscale = multiscale_blob_channels(frames)
        # Channel order is current/mean for (3,9), (5,13), and (7,15).
        self._safe_negative_evidence = 0.5 * (
            multiscale[:, 2:3] + multiscale[:, 3:4]
        )
        return torch.cat((learned, multiscale), dim=1)

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        output = super().forward(frames)
        evidence = self._safe_negative_evidence
        self._safe_negative_evidence = None
        if evidence is None:
            raise RuntimeError("safe-negative evidence was not produced")
        output["safe_negative_evidence"] = evidence.detach()
        return output
