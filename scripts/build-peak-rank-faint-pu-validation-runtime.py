#!/usr/bin/env python
"""Package the verified faint-cell capacity detector for clean validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-faint-pu-validation-runtime-v4"
module["main"].__globals__.update(
    {
        "STATE": ROOT / ".biohub" / "cache" / "antelume-peak-rank-faint-pu-v4",
        "ARCHIVE": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-faint-pu-v4"
        / "peak-rank-faint-pu-v4-results.tar.gz",
        "REPORT": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-faint-pu-v4"
        / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-real-conservative-pu-faint-temporal-peak-rank-v4"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Faint-PU Validation Runtime v4",
        "EXPECTED_PARAMETER_COUNT": 66_977_670,
    }
)


if __name__ == "__main__":
    module["main"]()

