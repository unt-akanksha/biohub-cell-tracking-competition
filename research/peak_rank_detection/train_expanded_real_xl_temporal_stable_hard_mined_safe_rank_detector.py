#!/usr/bin/env python
"""Train the XL temporal-SNR detector with frozen hard-example sampling."""

from __future__ import annotations

try:
    import hard_mined_base as hard_mined
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_temporal_stable_hard_mined_safe_rank_detector as hard_mined,
    )


RUN_ID = (
    "synthetic256-expanded-real-xl-temporal-min-balanced-hard-mined-pu-faint-"
    "local-shape-multiscale-blob-global-safe-rank-peak-rank-v31"
)


def main() -> None:
    hard_mined.RUN_ID = RUN_ID
    hard_mined.main()


if __name__ == "__main__":
    main()
