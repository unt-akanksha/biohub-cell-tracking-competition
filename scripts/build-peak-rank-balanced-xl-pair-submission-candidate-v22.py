#!/usr/bin/env python
"""Build the fixed balanced base/XL tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-balanced-xl-pair-tracking-candidate-v22"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-balanced-xl-pair-validation-"
            "runtime-v22"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-balanced-xl-pair-tracking-candidate-v22",
        "TARGET_TITLE": "Biohub Peak Rank Balanced XL Pair Candidate v22",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "213.6M-parameter fixed balanced base/XL ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
