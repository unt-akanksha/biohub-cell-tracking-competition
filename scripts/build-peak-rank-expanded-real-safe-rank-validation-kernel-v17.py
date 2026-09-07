#!/usr/bin/env python
"""Build dual-GPU validation for the safe-rank member."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-expanded-real-safe-rank-validation-v17"
runtime_slug = "biohub-peak-rank-expanded-real-safe-rank-validation-runtime-v17"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank Safe-Rank Validation v17",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 83_802_246,
    }
)


if __name__ == "__main__":
    module["main"]()
