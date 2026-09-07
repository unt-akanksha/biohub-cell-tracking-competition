#!/usr/bin/env python
"""Package the verified capacity-scaled PU detector for clean validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-capacity-pu-validation-runtime-v3"
module["main"].__globals__.update(
    {
        "STATE": ROOT / ".biohub" / "cache" / "antelume-peak-rank-capacity-pu-v3",
        "ARCHIVE": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-capacity-pu-v3"
        / "peak-rank-capacity-pu-v3-results.tar.gz",
        "REPORT": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-capacity-pu-v3"
        / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Capacity-PU Validation Runtime v3",
        "EXPECTED_PARAMETER_COUNT": 66_977_670,
    }
)


if __name__ == "__main__":
    module["main"]()
