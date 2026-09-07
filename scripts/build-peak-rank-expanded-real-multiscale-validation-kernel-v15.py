#!/usr/bin/env python
"""Build the dual-GPU validation kernel for the multiscale member."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-expanded-real-multiscale-validation-v15"
runtime_slug = "biohub-peak-rank-expanded-real-multiscale-validation-runtime-v15"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank Expanded-Real Multiscale Validation v15",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 83_802_246,
    }
)


if __name__ == "__main__":
    module["main"]()
