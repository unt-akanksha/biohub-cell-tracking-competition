#!/usr/bin/env python
"""Build the fixed safe-rank/embryo-balanced tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-safe-balanced-pair-tracking-candidate-v20"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-safe-balanced-pair-validation-"
            "runtime-v20"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-safe-balanced-pair-tracking-candidate-v20"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Safe Balanced Pair Candidate v20",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "167.6M-parameter fixed safe-rank/embryo-balanced ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
