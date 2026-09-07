#!/usr/bin/env python
"""Build private validation for the expanded-real detector trio."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-expanded-blob-ensemble-validation-v12"
runtime_slug = "biohub-peak-rank-expanded-blob-ensemble-validation-runtime-v12"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": "Biohub Peak Rank Expanded Blob Ensemble Validation v12",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 200_939_922,
    }
)


if __name__ == "__main__":
    module["main"]()
