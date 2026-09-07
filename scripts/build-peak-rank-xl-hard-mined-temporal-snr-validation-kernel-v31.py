#!/usr/bin/env python
"""Build dual-GPU validation for the V31 XL hard-mined detector."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-xl-hard-mined-temporal-snr-validation-v31"
runtime_slug = "biohub-peak-rank-xl-hard-mined-temporal-snr-validation-runtime-v31"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank XL Hard-Mined Temporal-SNR Validation v31",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 129_761_606,
    }
)


if __name__ == "__main__":
    module["main"]()
