#!/usr/bin/env python
"""Package the fixed expanded-real plus local-shape detector ensemble."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10"
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": (
            "Biohub Peak Rank Expanded Local-Shape Ensemble Runtime v10"
        ),
        "PURPOSE": (
            "Two-GPU clean validation of a fixed equal-logit ensemble of "
            "expanded-real faint-cell and local-shape detectors"
        ),
        "PARAMETER_COUNT": 133_955_340,
        "RUN_ID": "clean-expanded-local-shape-equal-logit-ensemble-v10",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed two-member equal-logit ensemble of independently gated "
            "expanded-real temporal 3D ConvNeXt U-Net peak rankers"
        ),
        "MEMBERS": (
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
            {
                "name": "expanded-real-local-shape-v9",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-expanded-real-local-shape-validation-controller-v9.json",
                "checkpoint_file": "member-expanded-real-local-shape-v9.pt",
                "expected_parameter_count": 66_977_670,
            },
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
