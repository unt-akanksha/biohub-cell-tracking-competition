#!/usr/bin/env python
"""Build the fixed expanded-real plus local-shape tracking candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-expanded-local-shape-ensemble-tracking-candidate-v10"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/"
            "biohub-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-expanded-local-shape-ensemble-tracking-candidate-v10"
        ),
        "TARGET_TITLE": (
            "Biohub Peak Rank Expanded Local-Shape Ensemble Candidate v10"
        ),
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter temporal peak-ranking detector",
            "two-member 134.0M-parameter expanded-real plus local-shape "
            "equal-logit temporal peak-ranking ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
