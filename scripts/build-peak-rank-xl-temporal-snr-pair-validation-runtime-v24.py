#!/usr/bin/env python
"""Package the fixed XL and temporal local-SNR detector pair."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-logit-ensemble-validation-runtime.py")
)
target_name = "biohub-peak-rank-xl-temporal-snr-pair-validation-runtime-v24"
sources = dict(module["main"].__globals__["SOURCES"])
for name in (
    "model_blob.py",
    "model_global.py",
    "model_multiscale.py",
    "model_safe_rank.py",
    "model_temporal_stable.py",
):
    sources[name] = ROOT / "research" / "peak_rank_detection" / name
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank XL Temporal-SNR Pair Runtime v24",
        "PURPOSE": (
            "Two-GPU clean validation of a precommitted equal-logit pair of "
            "individually accepted capacity- and evidence-diverse detectors"
        ),
        "PARAMETER_COUNT": 213_561_260,
        "RUN_ID": "clean-xl-temporal-snr-equal-logit-ensemble-v24",
        "ARCHITECTURE_DESCRIPTION": (
            "fixed two-member equal-logit ensemble of an XL multiscale global "
            "temporal 3D ConvNeXt U-Net and an independent temporal-minimum "
            "local-SNR temporal 3D ConvNeXt U-Net"
        ),
        "SOURCES": sources,
        "MEMBERS": (
            {
                "name": "expanded-real-xl-balanced-v21",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-expanded-real-xl-balanced-validation-runtime-v21",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-expanded-real-xl-balanced-validation-controller-v21.json",
                "checkpoint_file": "member-expanded-real-xl-balanced-v21.pt",
                "expected_parameter_count": 129_748_646,
            },
            {
                "name": "temporal-min-local-snr-v23",
                "runtime": ROOT
                / ".biohub"
                / "staging"
                / "biohub-peak-rank-temporal-min-local-snr-validation-runtime-v23",
                "controller": ROOT
                / ".biohub"
                / "automation"
                / "peak-rank-temporal-min-local-snr-validation-controller-v23.json",
                "checkpoint_file": "member-temporal-min-local-snr-v23.pt",
                "expected_parameter_count": 83_812_614,
            },
        ),
    }
)


if __name__ == "__main__":
    module["main"]()
