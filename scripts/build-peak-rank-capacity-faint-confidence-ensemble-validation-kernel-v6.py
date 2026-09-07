#!/usr/bin/env python
"""Build private validation for confidence-selective capacity+faint fusion."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-capacity-faint-confidence-validation-v6"
runtime_slug = "biohub-peak-rank-capacity-faint-confidence-validation-runtime-v6"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank Capacity Faint Confidence Validation v6",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 133_955_340,
    }
)


if __name__ == "__main__":
    module["main"]()
