#!/usr/bin/env python
"""Package the fixed capacity, faint, and expanded-real detector ensemble."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
wrapped = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-capacity-faint-ensemble-validation-runtime-v5.py")
)
module = wrapped["module"]
target_name = "biohub-peak-rank-cfe-ensemble-validation-runtime-v8"
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Capacity Faint Expanded Ensemble v8",
        "PURPOSE": (
            "Two-GPU clean validation of a fixed equal-logit ensemble of "
            "capacity, faint-cell, and expanded-real detectors"
        ),
        "PARAMETER_COUNT": 200_933_010,
        "RUN_ID": "clean-capacity-faint-expanded-equal-logit-ensemble-v8",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed three-member equal-logit ensemble of independently gated "
            "temporal 3D ConvNeXt U-Net peak rankers"
        ),
        "MEMBERS": (
            {
                "name": "capacity-pu-v3",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-capacity-pu-validation-runtime-v3",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-capacity-pu-validation-controller-v3.json",
                "checkpoint_file": "member-capacity-pu-v3.pt",
                "expected_parameter_count": 66_977_670,
            },
            {
                "name": "faint-pu-v4",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-faint-pu-validation-runtime-v4",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-faint-pu-validation-controller-v4.json",
                "checkpoint_file": "member-faint-pu-v4.pt",
                "expected_parameter_count": 66_977_670,
            },
            {
                "name": "expanded-real-faint-v7",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-expanded-real-faint-validation-runtime-v7",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-expanded-real-faint-validation-controller-v7.json",
                "checkpoint_file": "member-expanded-real-faint-v7.pt",
                "expected_parameter_count": 66_977_670,
            },
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
