#!/usr/bin/env python
"""Build the V27 hard-mined temporal-SNR tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-hard-mined-temporal-snr-tracking-candidate-v27"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-hard-mined-temporal-snr-validation-"
            "runtime-v27"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": "peak-rank-hard-mined-temporal-snr-tracking-candidate-v27",
        "TARGET_TITLE": "Biohub Peak Rank Hard-Mined Temporal-SNR Candidate v27",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "83.8M-parameter embryo-balanced, hard-mined temporal-SNR",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
