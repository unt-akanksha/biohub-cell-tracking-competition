#!/usr/bin/env python
"""Train the balanced safe-rank detector with temporal-minimum blob evidence."""

from __future__ import annotations

try:
    import balanced_safe_rank_base as balanced
    from model_temporal_stable import (
        MODEL_FAMILY,
        TemporalMinimumLocalSnrSafeRankDetector,
    )
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_balanced_safe_rank_detector as balanced,
    )
    from research.peak_rank_detection.model_temporal_stable import (
        MODEL_FAMILY,
        TemporalMinimumLocalSnrSafeRankDetector,
    )


RUN_ID = (
    "synthetic256-expanded-real-temporal-min-balanced-pu-faint-local-shape-"
    "multiscale-blob-global-safe-rank-peak-rank-v23"
)


def main() -> None:
    balanced.RUN_ID = RUN_ID
    balanced.safe.MODEL_FAMILY = MODEL_FAMILY
    balanced.safe.SafeRankMultiscaleBlobGlobalDetector = (
        TemporalMinimumLocalSnrSafeRankDetector
    )
    balanced.main()


if __name__ == "__main__":
    main()
