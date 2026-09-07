#!/usr/bin/env python
"""Package the fixed capacity+faint equal-logit ensemble for validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-capacity-faint-ensemble-validation-runtime-v5"
globals_ = module["main"].__globals__
globals_.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Capacity Faint Ensemble Runtime v5",
        "PURPOSE": "Two-GPU clean validation of a fixed capacity and faint-cell equal-logit detector ensemble",
        "PARAMETER_COUNT": 133_955_340,
        "RUN_ID": "clean-capacity-faint-equal-logit-peak-rank-ensemble-v5",
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
        ),
    }
)


if __name__ == "__main__":
    module["main"]()

