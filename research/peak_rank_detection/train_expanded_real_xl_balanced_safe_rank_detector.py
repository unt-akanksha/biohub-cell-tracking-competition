#!/usr/bin/env python
"""Train the XL safe-rank detector with balanced real sampling."""

from __future__ import annotations

try:
    import balanced_safe_rank_base as balanced
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_balanced_safe_rank_detector as balanced,
    )


RUN_ID = (
    "synthetic256-expanded-real-xl-balanced-pu-faint-local-shape-multiscale-"
    "blob-global-safe-rank-peak-rank-v21"
)


def main() -> None:
    balanced.RUN_ID = RUN_ID
    balanced.main()


if __name__ == "__main__":
    main()
