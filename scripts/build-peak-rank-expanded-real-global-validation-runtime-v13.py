#!/usr/bin/env python
"""Package the verified expanded-real global-context detector for validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-expanded-real-global-validation-runtime-v13"
state = ROOT / ".biohub" / "cache" / "antelume-peak-rank-expanded-real-global-v13"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
sources["model_global.py"] = ROOT / "research" / "peak_rank_detection" / "model_global.py"
module["main"].__globals__.update(
    {
        "STATE": state,
        "ARCHIVE": state / "peak-rank-expanded-real-global-v13-results.tar.gz",
        "REPORT": state / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-expanded-real-pu-faint-local-shape-blob-global-peak-rank-v13"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": (
            "Biohub Peak Rank Expanded-Real Global Validation Runtime v13"
        ),
        "EXPECTED_PARAMETER_COUNT": 83_788_422,
        "ARCHITECTURE_DESCRIPTION": (
            "independent blob-aware temporal 3D ConvNeXt U-Net peak ranker "
            "with two full-field bottleneck attention blocks"
        ),
        "SOURCES": sources,
    }
)


if __name__ == "__main__":
    module["main"]()
