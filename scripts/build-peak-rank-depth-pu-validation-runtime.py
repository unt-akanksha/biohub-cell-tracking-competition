#!/usr/bin/env python
"""Package the verified depth-robust PU detector for clean Kaggle validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-depth-pu-validation-runtime-v2"
module["main"].__globals__.update(
    {
        "STATE": ROOT / ".biohub" / "cache" / "antelume-peak-rank-depth-pu-v2",
        "ARCHIVE": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-depth-pu-v2"
        / "peak-rank-depth-pu-v2-results.tar.gz",
        "REPORT": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-depth-pu-v2"
        / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": "synthetic256-real-conservative-pu-depth-robust-peak-rank-v2",
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Depth-PU Validation Runtime v2",
    }
)


if __name__ == "__main__":
    module["main"]()
