#!/usr/bin/env python
"""Build the fixed v1+v2 equal-logit tracking submission candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-logit-ensemble-tracking-candidate-v4"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": "indarkarhana/biohub-peak-rank-logit-ensemble-validation-runtime-v4",
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-logit-ensemble-tracking-candidate-v4",
        "TARGET_TITLE": "Biohub Peak Rank Equal-Logit Ensemble Candidate v4",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter temporal peak-ranking detector",
            "two-member 76.8M-parameter equal-logit temporal peak-ranking ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
