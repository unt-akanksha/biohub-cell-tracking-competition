#!/usr/bin/env python
"""Train v17 with evidence-filtered negative ranking on sparse real crops."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    import local_shape_base as local_shape
    from model_safe_rank import MODEL_FAMILY, SafeRankMultiscaleBlobGlobalDetector
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_local_shape_detector as local_shape,
    )
    from research.peak_rank_detection.model_safe_rank import (
        MODEL_FAMILY,
        SafeRankMultiscaleBlobGlobalDetector,
    )


RUN_ID = (
    "synthetic256-expanded-real-pu-faint-local-shape-multiscale-blob-global-"
    "safe-rank-peak-rank-v17"
)
SAFE_SHELL_MINIMUM_RADIUS = 3.0
SAFE_SHELL_MAXIMUM_RADIUS = 8.0
SAFE_EXCLUSION_RADIUS = 2.5
SAFE_BACKGROUND_QUANTILE = 0.5
SAFE_RANK_MARGIN = 0.5
SAFE_RANK_HARD_NEGATIVES = 16
SAFE_RANK_WEIGHT = 0.25
OPTIMIZATION_ANNOTATED_BELOW_BOUNDARY_FRACTION = 0.02811804008908686

SAFE_SHELL_OFFSETS = torch.tensor(
    [
        offset
        for offset in product(range(-8, 9), repeat=3)
        if SAFE_SHELL_MINIMUM_RADIUS
        <= math.sqrt(sum(value * value for value in offset))
        <= SAFE_SHELL_MAXIMUM_RADIUS
    ],
    dtype=torch.long,
)


def safe_background_ranking_loss(
    logits: torch.Tensor,
    evidence: torch.Tensor,
    points: torch.Tensor,
    *,
    quantile: float = SAFE_BACKGROUND_QUANTILE,
    margin: float = SAFE_RANK_MARGIN,
    hard_negatives: int = SAFE_RANK_HARD_NEGATIVES,
) -> torch.Tensor:
    """Rank known centers over only dark, safely separated shell voxels."""

    if logits.ndim != 5 or logits.shape[:2] != (1, 1):
        raise ValueError("logits must have shape (1,1,Z,Y,X)")
    if evidence.shape != logits.shape:
        raise ValueError("evidence must have the same shape as logits")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if not 0.0 < quantile < 1.0 or margin <= 0.0 or hard_negatives <= 0:
        raise ValueError("safe-ranking configuration is invalid")
    if not len(points):
        return logits.sum() * 0.0

    spatial = torch.tensor(logits.shape[-3:], device=points.device)
    centers = points.round().long()
    valid = ((centers >= 0) & (centers < spatial)).all(dim=1)
    centers = centers[valid]
    known_points = points[valid]
    if not len(centers):
        return logits.sum() * 0.0

    boundary = torch.quantile(evidence.detach().float(), quantile)
    offsets = SAFE_SHELL_OFFSETS.to(device=points.device)
    losses: list[torch.Tensor] = []
    for center in centers:
        candidates = center[None] + offsets
        inside = ((candidates >= 0) & (candidates < spatial)).all(dim=1)
        candidates = candidates[inside]
        if not len(candidates):
            continue
        distances = torch.cdist(candidates.float(), known_points.float(), p=2)
        candidates = candidates[(distances > SAFE_EXCLUSION_RADIUS).all(dim=1)]
        if not len(candidates):
            continue
        candidate_evidence = evidence[
            0, 0, candidates[:, 0], candidates[:, 1], candidates[:, 2]
        ]
        candidates = candidates[candidate_evidence <= boundary]
        if not len(candidates):
            continue
        negative_logits = logits[
            0, 0, candidates[:, 0], candidates[:, 1], candidates[:, 2]
        ]
        hardest = negative_logits.topk(min(hard_negatives, len(negative_logits))).values
        positive = logits[0, 0, center[0], center[1], center[2]]
        losses.append(F.softplus(hardest - positive + margin).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def training_loss(
    prediction: dict[str, Any],
    points: torch.Tensor,
    *,
    complete_labels: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, components = local_shape.training_loss(
        prediction, points, complete_labels=complete_labels
    )
    if complete_labels:
        return loss, {**components, "safe_background_rank": 0.0}
    safe_rank = safe_background_ranking_loss(
        prediction["logits"], prediction["safe_negative_evidence"], points
    )
    return loss + SAFE_RANK_WEIGHT * safe_rank, {
        **components,
        "safe_background_rank": float(safe_rank.detach()),
    }


def tagged_atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    writer: Callable[[Path, dict[str, Any]], None],
) -> None:
    tagged = dict(payload)
    if path.name == "terminal.json":
        tagged.update(
            {
                "model_family": MODEL_FAMILY,
                "safe_negative_evidence_band": [5, 13],
                "safe_negative_boundary_quantile": SAFE_BACKGROUND_QUANTILE,
                "safe_negative_shell_radius_voxels": [
                    SAFE_SHELL_MINIMUM_RADIUS,
                    SAFE_SHELL_MAXIMUM_RADIUS,
                ],
                "safe_negative_selection_role": "expanded_real_optimization_only",
                "optimization_annotated_below_boundary_fraction": (
                    OPTIMIZATION_ANNOTATED_BELOW_BOUNDARY_FRACTION
                ),
            }
        )
    writer(path, tagged)


def main() -> None:
    base = local_shape.expanded.base
    original_writer = base.atomic_json

    def write_with_family(path: Path, payload: dict[str, Any]) -> None:
        tagged_atomic_json(path, payload, writer=original_writer)

    base.RUN_ID = RUN_ID
    base.TemporalPeakRankDetector = SafeRankMultiscaleBlobGlobalDetector
    base.validate_real_manifest = (
        local_shape.expanded.validate_expanded_real_manifest
    )
    base.augment_example = local_shape.faint.augment_example
    base.training_loss = training_loss
    base.atomic_json = write_with_family
    base.main()


if __name__ == "__main__":
    main()
