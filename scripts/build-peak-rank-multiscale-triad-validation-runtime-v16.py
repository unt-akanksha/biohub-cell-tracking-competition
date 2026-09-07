#!/usr/bin/env python
"""Package the fixed local/global/multiscale detector triad."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-multiscale-triad-validation-runtime-v16"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
sources["model_global.py"] = ROOT / "research" / "peak_rank_detection" / "model_global.py"
sources["model_multiscale.py"] = (
    ROOT / "research" / "peak_rank_detection" / "model_multiscale.py"
)
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Multiscale Triad Runtime v16",
        "PURPOSE": (
            "Two-GPU clean validation of a precommitted equal-logit ensemble "
            "of local blob, global-context, and multiscale-global detectors"
        ),
        "PARAMETER_COUNT": 234_575_250,
        "RUN_ID": "clean-local-global-multiscale-equal-logit-ensemble-v16",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed three-member equal-logit ensemble of independently gated "
            "expanded-real temporal 3D peak rankers"
        ),
        "SOURCES": sources,
        "MEMBERS": (
            {
                "name": "expanded-real-blob-v11",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-expanded-real-blob-validation-runtime-v11",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-expanded-real-blob-validation-controller-v11.json",
                "checkpoint_file": "member-expanded-real-blob-v11.pt",
                "expected_parameter_count": 66_984_582,
            },
            {
                "name": "expanded-real-global-v13",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-expanded-real-global-validation-runtime-v13",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-expanded-real-global-validation-controller-v13.json",
                "checkpoint_file": "member-expanded-real-global-v13.pt",
                "expected_parameter_count": 83_788_422,
            },
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
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
