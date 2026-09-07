#!/usr/bin/env python
"""Build the confidence-selective capacity+faint tracking candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-capacity-faint-confidence-tracking-candidate-v6"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": "indarkarhana/biohub-peak-rank-capacity-faint-confidence-validation-runtime-v6",
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-capacity-faint-confidence-tracking-candidate-v6",
        "TARGET_TITLE": "Biohub Peak Rank Capacity Faint Confidence Candidate v6",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter temporal peak-ranking detector",
            "two-member 134.0M-parameter confidence-selective capacity plus faint-cell temporal peak-ranking ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
