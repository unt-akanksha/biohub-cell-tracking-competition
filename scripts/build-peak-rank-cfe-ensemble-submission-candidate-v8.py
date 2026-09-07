#!/usr/bin/env python
"""Build the fixed capacity, faint, and expanded-real tracking candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-cfe-ensemble-tracking-candidate-v8"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-cfe-ensemble-validation-runtime-v8"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-cfe-ensemble-tracking-candidate-v8",
        "TARGET_TITLE": "Biohub Peak Rank Capacity Faint Expanded Candidate v8",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "200.9M-parameter fixed capacity/faint/expanded-real ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
