#!/usr/bin/env python
"""Build the fixed local/global/multiscale triad tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-multiscale-triad-tracking-candidate-v16"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-multiscale-triad-validation-runtime-v16"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-multiscale-triad-tracking-candidate-v16",
        "TARGET_TITLE": "Biohub Peak Rank Multiscale Triad Candidate v16",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "234.6M-parameter fixed local/global/multiscale ensemble",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
