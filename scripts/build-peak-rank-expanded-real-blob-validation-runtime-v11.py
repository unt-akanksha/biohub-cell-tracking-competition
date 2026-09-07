#!/usr/bin/env python
"""Package the verified expanded-real blob-aware detector for validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-expanded-real-blob-validation-runtime-v11"
state = ROOT / ".biohub" / "cache" / "antelume-peak-rank-expanded-real-blob-v11"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
module["main"].__globals__.update(
    {
        "STATE": state,
        "ARCHIVE": state / "peak-rank-expanded-real-blob-v11-results.tar.gz",
        "REPORT": state / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-expanded-real-pu-faint-local-shape-blob-peak-rank-v11"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": (
            "Biohub Peak Rank Expanded-Real Blob Validation Runtime v11"
        ),
        "EXPECTED_PARAMETER_COUNT": 66_984_582,
        "ARCHITECTURE_DESCRIPTION": (
            "independent blob-aware temporal 3D ConvNeXt U-Net peak ranker "
            "with fixed train-selected multiscale local-contrast channels"
        ),
        "SOURCES": sources,
    }
)


if __name__ == "__main__":
    module["main"]()
