#!/usr/bin/env python
"""Build private validation for the fixed safe-rank balanced pair."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-safe-balanced-pair-validation-v20"
runtime_slug = "biohub-peak-rank-safe-balanced-pair-validation-runtime-v20"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank Safe Balanced Pair Validation v20",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 167_604_492,
    }
)


if __name__ == "__main__":
    module["main"]()
