#!/usr/bin/env python
"""Package the fixed blob-aware and global-context detector pair."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-blob-global-ensemble-validation-runtime-v14"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
sources["model_global.py"] = ROOT / "research" / "peak_rank_detection" / "model_global.py"
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Blob Global Ensemble Runtime v14",
        "PURPOSE": (
            "Two-GPU clean validation of a fixed equal-logit ensemble of "
            "blob-aware local and global-context expanded-real detectors"
        ),
        "PARAMETER_COUNT": 150_773_004,
        "RUN_ID": "clean-blob-global-context-equal-logit-ensemble-v14",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed two-member equal-logit ensemble of independently gated "
            "blob-aware temporal 3D ConvNeXt U-Net peak rankers"
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
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
