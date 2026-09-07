#!/usr/bin/env python
"""Build the fixed expanded-real detector-trio tracking candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-expanded-blob-ensemble-tracking-candidate-v12"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-expanded-blob-ensemble-validation-runtime-v12"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-expanded-blob-ensemble-tracking-candidate-v12"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Expanded Blob Ensemble Candidate v12",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "200.9M-parameter fixed expanded-real faint/local-shape/blob ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
