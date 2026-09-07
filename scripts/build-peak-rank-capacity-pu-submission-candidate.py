#!/usr/bin/env python
"""Build the distinct capacity-scaled PU tracking submission candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-capacity-pu-tracking-candidate-v3"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": "indarkarhana/biohub-peak-rank-capacity-pu-validation-runtime-v3",
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-capacity-pu-tracking-candidate-v3",
        "TARGET_TITLE": "Biohub Peak Rank Capacity-PU Tracking Candidate v3",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter", "67.0M-parameter capacity-scaled"
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
