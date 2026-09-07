#!/usr/bin/env python
"""Package the verified expanded-real multiscale detector for validation."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-expanded-real-multiscale-validation-runtime-v15"
state = ROOT / ".biohub" / "cache" / "antelume-peak-rank-expanded-real-multiscale-v15"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
sources["model_global.py"] = ROOT / "research" / "peak_rank_detection" / "model_global.py"
sources["model_multiscale.py"] = (
    ROOT / "research" / "peak_rank_detection" / "model_multiscale.py"
)
module["main"].__globals__.update(
    {
        "STATE": state,
        "ARCHIVE": state / "peak-rank-expanded-real-multiscale-v15-results.tar.gz",
        "REPORT": state / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-expanded-real-pu-faint-local-shape-multiscale-blob-"
            "global-peak-rank-v15"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": (
            "Biohub Peak Rank Expanded-Real Multiscale Validation Runtime v15"
        ),
        "EXPECTED_PARAMETER_COUNT": 83_802_246,
        "ARCHITECTURE_DESCRIPTION": (
            "independent multi-scale blob-aware temporal 3D ConvNeXt U-Net "
            "with two full-field bottleneck attention blocks"
        ),
        "SOURCES": sources,
    }
)


if __name__ == "__main__":
    module["main"]()
