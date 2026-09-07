#!/usr/bin/env python
"""Train the capacity-scaled detector with conservative PU adaptation.

This precommitted third member keeps complete synthetic supervision, removes
false-negative pressure from sparsely annotated real voxels, and adds axial
attenuation.  Its architecture is selected by the runner through wider
ConvNeXt stages; this module owns only the data and loss policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from research.peak_rank_detection import train_synthetic_real_detector as base


RUN_ID = "synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3"
DEPTH_ATTENUATION_PROBABILITY = 0.75
MINIMUM_DEPTH_FACTOR = 0.25

_base_augment_example = base.augment_example
_base_training_loss = base.training_loss


def augment_example(
    example: base.TrainingExample,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    frames, points = _base_augment_example(example, rng)
    if rng.random() < DEPTH_ATTENUATION_PROBABILITY:
        depth_factor = float(rng.uniform(MINIMUM_DEPTH_FACTOR, 1.0))
        attenuation = np.linspace(
            1.0, depth_factor, frames.shape[1], dtype=np.float32
        )
        if rng.random() < 0.5:
            attenuation = attenuation[::-1].copy()
        frames = frames * attenuation[None, :, None, None]
    return np.ascontiguousarray(frames), points


def training_loss(
    prediction: dict[str, Any],
    points: torch.Tensor,
    *,
    complete_labels: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if complete_labels:
        return _base_training_loss(prediction, points, complete_labels=True)

    logits = prediction["logits"]
    heatmap = base.positive_logit_loss(logits, points)
    offset = base.subvoxel_offset_loss(prediction["offsets"], [points])
    auxiliary = logits.sum() * 0.0
    for auxiliary_logits in prediction["auxiliary_logits"]:
        scaled = points * (auxiliary_logits.shape[-1] / float(logits.shape[-1]))
        auxiliary = auxiliary + base.positive_logit_loss(auxiliary_logits, scaled)
    rank = logits.sum() * 0.0
    total = heatmap + 0.20 * offset + 0.10 * auxiliary
    return total, {
        "heatmap": float(heatmap.detach()),
        "rank": float(rank.detach()),
        "offset": float(offset.detach()),
        "auxiliary": float(auxiliary.detach()),
    }


def main() -> None:
    base.RUN_ID = RUN_ID
    base.augment_example = augment_example
    base.training_loss = training_loss
    base.main()


if __name__ == "__main__":
    main()
