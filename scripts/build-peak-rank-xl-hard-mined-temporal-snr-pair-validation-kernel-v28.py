#!/usr/bin/env python
"""Build validation for the XL/hard-mined temporal-SNR pair."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-xl-hard-mined-temporal-snr-pair-validation-v28"
runtime_slug = "biohub-peak-rank-xl-hard-mined-temporal-snr-pair-validation-runtime-v28"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank XL Hard-Mined Temporal-SNR Pair v28",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 213_561_260,
    }
)


if __name__ == "__main__":
    module["main"]()
