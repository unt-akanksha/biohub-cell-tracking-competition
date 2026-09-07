#!/usr/bin/env python
"""Package the fixed multiscale baseline and safe-rank detector pair."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-multiscale-safe-pair-validation-runtime-v18"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
sources["model_global.py"] = ROOT / "research" / "peak_rank_detection" / "model_global.py"
sources["model_multiscale.py"] = (
    ROOT / "research" / "peak_rank_detection" / "model_multiscale.py"
)
sources["model_safe_rank.py"] = (
    ROOT / "research" / "peak_rank_detection" / "model_safe_rank.py"
)
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Multiscale Safe Pair Runtime v18",
        "PURPOSE": (
            "Two-GPU clean validation of a precommitted equal-logit pair of "
            "multiscale baseline and evidence-filtered safe-rank detectors"
        ),
        "PARAMETER_COUNT": 167_604_492,
        "RUN_ID": "clean-multiscale-safe-rank-equal-logit-ensemble-v18",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed two-member equal-logit ensemble of independently trained "
            "multiscale global temporal 3D peak rankers"
        ),
        "SOURCES": sources,
        "MEMBERS": (
            {
                "name": "expanded-real-multiscale-v15",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-expanded-real-multiscale-validation-runtime-v15",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-expanded-real-multiscale-validation-controller-v15.json",
                "checkpoint_file": "member-expanded-real-multiscale-v15.pt",
                "expected_parameter_count": 83_802_246,
            },
            {
                "name": "expanded-real-safe-rank-v17",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-expanded-real-safe-rank-validation-runtime-v17",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-expanded-real-safe-rank-validation-controller-v17.json",
                "checkpoint_file": "member-expanded-real-safe-rank-v17.pt",
                "expected_parameter_count": 83_802_246,
            },
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
