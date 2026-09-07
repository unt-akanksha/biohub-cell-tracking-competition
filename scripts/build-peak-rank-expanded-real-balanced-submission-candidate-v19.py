#!/usr/bin/env python
"""Build the embryo-balanced safe-rank tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-expanded-real-balanced-tracking-candidate-v19"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-expanded-real-balanced-validation-"
            "runtime-v19"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-expanded-real-balanced-tracking-candidate-v19"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Expanded-Real Balanced Candidate v19",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "83.8M-parameter embryo-balanced multiscale safe-rank",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
