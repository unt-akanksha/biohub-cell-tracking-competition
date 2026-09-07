#!/usr/bin/env python
"""Package the verified XL embryo-balanced detector for validation."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-expanded-real-xl-balanced-validation-runtime-v21"
state = ROOT / ".biohub" / "cache" / "antelume-peak-rank-expanded-real-xl-balanced-v21"
sources = dict(module["main"].__globals__["SOURCES"])
sources["model_blob.py"] = ROOT / "research" / "peak_rank_detection" / "model_blob.py"
sources["model_global.py"] = ROOT / "research" / "peak_rank_detection" / "model_global.py"
sources["model_multiscale.py"] = ROOT / "research" / "peak_rank_detection" / "model_multiscale.py"
sources["model_safe_rank.py"] = ROOT / "research" / "peak_rank_detection" / "model_safe_rank.py"
module["main"].__globals__.update(
    {
        "STATE": state,
        "ARCHIVE": state / "peak-rank-expanded-real-xl-balanced-v21-results.tar.gz",
        "REPORT": state / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-expanded-real-xl-balanced-pu-faint-local-shape-"
            "multiscale-blob-global-safe-rank-peak-rank-v21"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank XL Balanced Runtime v21",
        "EXPECTED_PARAMETER_COUNT": 129_748_646,
        "ARCHITECTURE_DESCRIPTION": (
            "129.7M-parameter multiscale global temporal 3D ConvNeXt U-Net "
            "with evidence-filtered ranking and embryo-balanced optimization"
        ),
        "SOURCES": sources,
    }
)


if __name__ == "__main__":
    module["main"]()
