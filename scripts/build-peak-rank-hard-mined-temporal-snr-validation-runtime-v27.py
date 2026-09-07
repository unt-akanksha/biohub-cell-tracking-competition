#!/usr/bin/env python
"""Package the verified V27 hard-mined temporal-SNR detector."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
target_name = "biohub-peak-rank-hard-mined-temporal-snr-validation-runtime-v27"
state = ROOT / ".biohub" / "cache" / "antelume-peak-rank-hard-mined-temporal-snr-v27"
sources = dict(module["main"].__globals__["SOURCES"])
for filename in (
    "model_blob.py",
    "model_global.py",
    "model_multiscale.py",
    "model_safe_rank.py",
    "model_temporal_stable.py",
):
    sources[filename] = ROOT / "research" / "peak_rank_detection" / filename
module["main"].__globals__.update(
    {
        "STATE": state,
        "ARCHIVE": state / "peak-rank-hard-mined-temporal-snr-v27-results.tar.gz",
        "REPORT": state / "harvest-verification.json",
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "ARCHIVE_ROOT": (
            "synthetic256-expanded-real-temporal-min-balanced-hard-mined-pu-"
            "faint-local-shape-multiscale-blob-global-safe-rank-peak-rank-v27"
        ),
        "EXPECTED_TARGET_NAME": target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Hard-Mined Temporal-SNR Runtime v27",
        "EXPECTED_PARAMETER_COUNT": 83_812_614,
        "ARCHITECTURE_DESCRIPTION": (
            "independent multiscale blob/global temporal 3D ConvNeXt U-Net "
            "with temporal-minimum local-SNR evidence, safe ranking, embryo "
            "balance, and optimization-only hard-example sampling"
        ),
        "SOURCES": sources,
    }
)


if __name__ == "__main__":
    module["main"]()
