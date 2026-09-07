#!/usr/bin/env python
"""Build the distinct expanded-real local-shape tracking candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-expanded-real-local-shape-tracking-candidate-v9"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/"
            "biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-expanded-real-local-shape-tracking-candidate-v9"
        ),
        "TARGET_TITLE": (
            "Biohub Peak Rank Expanded-Real Local-Shape Tracking Candidate v9"
        ),
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "67.0M-parameter expanded-real local-shape faint-cell",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
