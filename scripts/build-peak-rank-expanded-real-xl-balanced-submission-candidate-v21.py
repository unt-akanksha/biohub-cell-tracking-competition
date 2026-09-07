#!/usr/bin/env python
"""Build the XL embryo-balanced safe-rank tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-expanded-real-xl-balanced-tracking-candidate-v21"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-expanded-real-xl-balanced-"
            "validation-runtime-v21"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-expanded-real-xl-balanced-tracking-candidate-v21"
        ),
        "TARGET_TITLE": "Biohub Peak Rank XL Balanced Candidate v21",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "129.7M-parameter XL embryo-balanced multiscale safe-rank",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
