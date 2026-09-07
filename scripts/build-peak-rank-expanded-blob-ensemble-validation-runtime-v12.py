#!/usr/bin/env python
"""Package the fixed expanded-real detector trio for clean validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-expanded-blob-ensemble-validation-runtime-v12"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Expanded Blob Ensemble Runtime v12",
        "PURPOSE": (
            "Two-GPU clean validation of a fixed equal-logit ensemble of "
            "expanded-real faint, local-shape, and blob-aware detectors"
        ),
        "PARAMETER_COUNT": 200_939_922,
        "RUN_ID": "clean-expanded-faint-local-shape-blob-equal-logit-ensemble-v12",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed three-member equal-logit ensemble of independently gated "
            "expanded-real temporal 3D ConvNeXt U-Net peak rankers"
        ),
        "SOURCES": sources,
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
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
