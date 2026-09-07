#!/usr/bin/env python
"""Package confidence-selective capacity+faint fusion for clean validation."""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
wrapped = runpy.run_path(
    str(ROOT / "scripts" / "build-peak-rank-capacity-faint-ensemble-validation-runtime-v5.py")
)
module = wrapped["module"]
target_name = "biohub-peak-rank-capacity-faint-confidence-validation-runtime-v6"
module["main"].__globals__.update(
    {
        "TARGET_NAME": target_name,
        "TARGET": ROOT / ".biohub" / "staging" / target_name,
        "DATASET_ID": f"indarkarhana/{target_name}",
        "DATASET_TITLE": "Biohub Peak Rank Capacity Faint Confidence Runtime v6",
        "PURPOSE": "Two-GPU clean validation of confidence-selective capacity and faint-cell detector fusion",
        "RUN_ID": "clean-capacity-faint-confidence-peak-rank-ensemble-v6",
        "FUSION": "confidence_max_logit_with_winner_offset",
        "ARCHITECTURE_DESCRIPTION": "confidence-selective ensemble of independent temporal 3D ConvNeXt U-Net peak rankers",
    }
)


if __name__ == "__main__":
    module["main"]()
