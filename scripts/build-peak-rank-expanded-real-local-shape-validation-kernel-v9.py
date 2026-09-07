#!/usr/bin/env python
"""Build the dual-GPU validation kernel for the local-shape member."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-kernel.py"))
target_id = "biohub-peak-rank-expanded-real-local-shape-validation-v9"
runtime_slug = "biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9"
module["main"].__globals__.update(
    {
        "TARGET_ID": target_id,
        "RUNTIME_SLUG": runtime_slug,
        "RUNTIME_REF": f"indarkarhana/{runtime_slug}",
        "KERNEL_TITLE": (
            "Biohub Peak Rank Expanded-Real Local-Shape Validation v9"
        ),
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "EXPECTED_PARAMETER_COUNT": 66_977_670,
    }
)


if __name__ == "__main__":
    module["main"]()
