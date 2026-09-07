#!/usr/bin/env python
"""Build the distinct depth-robust PU tracking submission candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-depth-pu-tracking-candidate-v2"
module["main"].__globals__.update(
    {
        "RUNTIME_REF": "indarkarhana/biohub-peak-rank-depth-pu-validation-runtime-v2",
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-depth-pu-tracking-candidate-v2",
        "TARGET_TITLE": "Biohub Peak Rank Depth-PU Tracking Candidate v2",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
    }
)


if __name__ == "__main__":
    module["main"]()
