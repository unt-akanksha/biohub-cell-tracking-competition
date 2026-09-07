#!/usr/bin/env python
"""Package the verified expanded-real local-shape detector for validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9"
state = ROOT / ".biohub" / "cache" / "antelume-peak-rank-expanded-real-local-shape-v9"
module["main"].__globals__.update(
    {
        "STATE": state,
        "ARCHIVE": state / "peak-rank-expanded-real-local-shape-v9-results.tar.gz",
        "REPORT": state / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-expanded-real-pu-faint-local-shape-peak-rank-v9"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": (
            "Biohub Peak Rank Expanded-Real Local-Shape Validation Runtime v9"
        ),
        "EXPECTED_PARAMETER_COUNT": 66_977_670,
        "ARCHITECTURE_DESCRIPTION": (
            "independent 67M temporal 3D ConvNeXt U-Net peak ranker trained "
            "with expanded real coverage, faint-cell augmentation, and a "
            "conservative annotated-center local-shape margin"
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
