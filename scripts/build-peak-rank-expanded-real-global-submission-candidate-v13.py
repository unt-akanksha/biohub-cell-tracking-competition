#!/usr/bin/env python
"""Build the distinct expanded-real global-context tracking candidate."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-submission-candidate.py"))
target_id = "biohub-peak-rank-expanded-real-global-tracking-candidate-v13"
globals_ = module["main"].__globals__
globals_.update(
    {
        "RUNTIME_REF": (
            "indarkarhana/biohub-peak-rank-expanded-real-global-validation-runtime-v13"
        ),
        "TARGET_ID": target_id,
        "CANDIDATE_RUN_ID": (
            "peak-rank-expanded-real-global-tracking-candidate-v13"
        ),
        "TARGET_TITLE": "Biohub Peak Rank Expanded-Real Global Candidate v13",
        "TARGET": ROOT / "kaggle" / target_id,
        "NOTEBOOK": ROOT / "kaggle" / target_id / f"{target_id}.ipynb",
        "ATTRIBUTION": globals_["ATTRIBUTION"].replace(
            "38.4M-parameter",
            "83.8M-parameter expanded-real blob/global-context local-shape",
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
