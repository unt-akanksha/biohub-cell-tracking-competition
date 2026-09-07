#!/usr/bin/env python
"""Build dual-GPU validation for the V27 hard-mined detector."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-hard-mined-temporal-snr-validation-v27"
runtime_slug = "biohub-peak-rank-hard-mined-temporal-snr-validation-runtime-v27"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank Hard-Mined Temporal-SNR Validation v27",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 83_812_614,
    }
)


if __name__ == "__main__":
    module["main"]()
