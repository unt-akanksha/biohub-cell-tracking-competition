#!/usr/bin/env python
"""Build the fixed multiscale baseline/safe-rank tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-multiscale-safe-pair-tracking-candidate-v18"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-multiscale-safe-pair-validation-"
            "runtime-v18"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-multiscale-safe-pair-tracking-candidate-v18"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Multiscale Safe Pair Candidate v18",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "167.6M-parameter fixed multiscale baseline/safe-rank ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
