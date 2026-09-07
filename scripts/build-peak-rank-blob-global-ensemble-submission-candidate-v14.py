#!/usr/bin/env python
"""Build the fixed blob/global detector-pair tracking candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-blob-global-ensemble-tracking-candidate-v14"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-blob-global-ensemble-validation-runtime-v14"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-blob-global-ensemble-tracking-candidate-v14"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Blob Global Ensemble Candidate v14",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "150.8M-parameter fixed blob/global-context ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
