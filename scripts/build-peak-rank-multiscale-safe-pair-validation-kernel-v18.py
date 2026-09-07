#!/usr/bin/env python
"""Build private validation for the fixed multiscale safe-rank pair."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-multiscale-safe-pair-validation-v18"
runtime_slug = "biohub-peak-rank-multiscale-safe-pair-validation-runtime-v18"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank Multiscale Safe Pair Validation v18",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 167_604_492,
    }
)


if __name__ == "__main__":
    module["main"]()
