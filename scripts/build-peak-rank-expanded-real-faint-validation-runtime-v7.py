#!/usr/bin/env python
"""Package the verified expanded-real faint detector for clean validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-expanded-real-faint-validation-runtime-v7"
module["main"].__globals__.update(
    {
        "STATE": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-expanded-real-faint-v7",
        "ARCHIVE": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-expanded-real-faint-v7"
        / "peak-rank-expanded-real-faint-v7-results.tar.gz",
        "REPORT": ROOT
        / ".biohub"
        / "cache"
        / "antelume-peak-rank-expanded-real-faint-v7"
        / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-expanded-real-pu-faint-temporal-peak-rank-v7"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Expanded-Real Faint Validation Runtime v7",
        "EXPECTED_PARAMETER_COUNT": 66_977_670,
        "ARCHITECTURE_DESCRIPTION": (
            "independent 67M temporal 3D ConvNeXt U-Net peak ranker trained with "
            "expanded clean real localization coverage and faint-cell augmentation"
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
