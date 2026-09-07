#!/usr/bin/env python
"""Train a conservative-PU detector with temporal faint-cell augmentation.

The augmentation removes local center-frame evidence while retaining adjacent
temporal context.  It is a generic training perturbation motivated by transient
fluorescence loss; it contains no movie identity, validation coordinate, public
prediction, or leaderboard-selected value.  Synthetic examples retain complete
supervision and sparse real examples contribute positive and offset loss only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from research.peak_rank_detection import train_synthetic_real_detector as base


RUN_ID = "synthetic256-real-conservative-pu-faint-temporal-peak-rank-v4"
DEPTH_ATTENUATION_PROBABILITY = 0.75
MINIMUM_DEPTH_FACTOR = 0.25
SYNTHETIC_FADE_PROBABILITY = 0.70
REAL_FADE_PROBABILITY = 0.40
MINIMUM_FADE_FRACTION = 0.12
MAXIMUM_FADE_FRACTION = 0.35
MAXIMUM_FADED_POINTS = 24
ADJACENT_FADE_PROBABILITY = 0.35

_base_augment_example = base.augment_example
_base_training_loss = base.training_loss


def attenuate_local_sphere(
    frame: np.ndarray,
    point: np.ndarray,
    *,
    center_factor: float,
    sigma: float,
) -> None:
    """Apply a soft multiplicative attenuation around one isotropic point."""

    if frame.ndim != 3:
        raise ValueError("frame must be three-dimensional")
    if not (0.0 <= center_factor <= 1.0) or sigma <= 0.0:
        raise ValueError("invalid attenuation parameters")
    center = np.asarray(point, dtype=np.float32).reshape(3)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    starts = np.maximum(0, np.floor(center).astype(np.int64) - radius)
    stops = np.minimum(
        np.asarray(frame.shape, dtype=np.int64),
        np.floor(center).astype(np.int64) + radius + 1,
    )
    if np.any(starts >= stops):
        return
    zz, yy, xx = np.ogrid[
        starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]
    ]
    distance_squared = (
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    )
    strength = np.exp(-0.5 * distance_squared / float(sigma * sigma)).astype(
        np.float32
    )
    multiplier = 1.0 - (1.0 - float(center_factor)) * strength
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
    frame[slices] *= multiplier


def apply_temporal_fading(
    frames: np.ndarray,
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float,
) -> np.ndarray:
    """Fade a stable random subset at t, sometimes extending into one neighbor."""

    values = np.ascontiguousarray(frames, dtype=np.float32).copy()
    if not len(points) or rng.random() >= probability:
        return values
    fraction = float(rng.uniform(MINIMUM_FADE_FRACTION, MAXIMUM_FADE_FRACTION))
    count = min(
        MAXIMUM_FADED_POINTS,
        len(points),
        max(1, int(math.ceil(len(points) * fraction))),
    )
    indices = np.asarray(rng.choice(len(points), size=count, replace=False)).reshape(-1)
    for index in indices:
        factor = float(rng.uniform(0.03, 0.35))
        sigma = float(rng.uniform(1.5, 3.0))
        attenuate_local_sphere(
            values[1], points[int(index)], center_factor=factor, sigma=sigma
        )
        if rng.random() < ADJACENT_FADE_PROBABILITY:
            neighbor = 0 if rng.random() < 0.5 else 2
            attenuate_local_sphere(
                values[neighbor],
                points[int(index)],
                center_factor=min(0.55, math.sqrt(factor)),
                sigma=sigma + 0.5,
            )
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


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
    probability = (
        SYNTHETIC_FADE_PROBABILITY
        if example.source == "synthetic"
        else REAL_FADE_PROBABILITY
    )
    frames = apply_temporal_fading(frames, points, rng, probability=probability)
    return np.ascontiguousarray(frames), np.ascontiguousarray(points)


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

