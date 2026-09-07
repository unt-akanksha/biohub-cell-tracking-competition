#!/usr/bin/env python
"""Build the temporal-minimum local-SNR tracking candidate."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-temporal-min-local-snr-tracking-candidate-v23"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-temporal-min-local-snr-validation-"
            "runtime-v23"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-temporal-min-local-snr-tracking-candidate-v23"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Temporal-Min Local-SNR Candidate v23",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "83.8M-parameter embryo-balanced temporal-minimum local-SNR",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
