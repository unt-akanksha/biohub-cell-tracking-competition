#!/usr/bin/env python
"""Package the fixed balanced base and XL detector pair."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-balanced-xl-pair-validation-runtime-v22"
sources = dict(module["main"].__globals__["SOURCES"])
for name in ("model_blob.py", "model_global.py", "model_multiscale.py", "model_safe_rank.py"):
    sources[name] = ROOT / "research" / "peak_rank_detection" / name
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Balanced XL Pair Runtime v22",
        "PURPOSE": (
            "Two-GPU clean validation of a precommitted equal-logit pair of "
            "independently seeded 83.8M and 129.7M balanced detectors"
        ),
        "PARAMETER_COUNT": 213_550_892,
        "RUN_ID": "clean-balanced-xl-equal-logit-ensemble-v22",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed two-member capacity-diverse equal-logit ensemble of "
            "evidence-filtered multiscale global temporal 3D peak rankers"
        ),
        "SOURCES": sources,
        "MEMBERS": (
            {
                "name": "expanded-real-balanced-v19",
                "runtime": ROOT / ".biohub" / "staging" / "biohub-peak-rank-expanded-real-balanced-validation-runtime-v19",
                "controller": ROOT / ".biohub" / "automation" / "peak-rank-expanded-real-balanced-validation-controller-v19.json",
                "checkpoint_file": "member-expanded-real-balanced-v19.pt",
                "expected_parameter_count": 83_802_246,
            },
            {
                "name": "expanded-real-xl-balanced-v21",
                "runtime": ROOT / ".biohub" / "staging" / "biohub-peak-rank-expanded-real-xl-balanced-validation-runtime-v21",
                "controller": ROOT / ".biohub" / "automation" / "peak-rank-expanded-real-xl-balanced-validation-controller-v21.json",
                "checkpoint_file": "member-expanded-real-xl-balanced-v21.pt",
                "expected_parameter_count": 129_748_646,
            },
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
