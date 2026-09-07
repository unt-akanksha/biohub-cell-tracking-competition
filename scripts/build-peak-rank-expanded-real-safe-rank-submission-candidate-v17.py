#!/usr/bin/env python
"""Build the distinct expanded-real safe-rank tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-expanded-real-safe-rank-tracking-candidate-v17"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-expanded-real-safe-rank-validation-"
            "runtime-v17"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-expanded-real-safe-rank-tracking-candidate-v17"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Expanded-Real Safe-Rank Candidate v17",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "83.8M-parameter expanded-real multiscale safe-rank",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
