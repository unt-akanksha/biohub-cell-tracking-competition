#!/usr/bin/env python
"""Train the expanded-real detector with conservative local peak shaping.

Competition-train crops remain positive-unlabeled: voxels across the volume
are never declared background.  Around each annotated center, however, a
small fixed shell is known not to be that same center.  A pairwise margin on
that shell teaches a unique local maximum without importing public weights,
predictions, test data, or leaderboard-selected constants.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import torch
import torch.nn.functional as F

try:
    # The remote queue uses flat, hash-pinned source copies so it does not
    # depend on a mutable checkout while waiting behind earlier members.
    import expanded_real_base as expanded
    import train_faint_cell_pu_detector as faint
except ModuleNotFoundError:
    from research.peak_rank_detection import (
        train_expanded_real_faint_detector as expanded,
    )
    from research.peak_rank_detection import train_faint_cell_pu_detector as faint


RUN_ID = "synthetic256-expanded-real-pu-faint-local-shape-peak-rank-v9"
LOCAL_SHELL_STEP = 2
LOCAL_SHAPE_MARGIN = 0.5
LOCAL_SHAPE_WEIGHT = 0.35

# Twenty-six directions at a two-voxel displacement.  Known neighboring
# centers are excluded below, so divisions and dense regions are protected.
LOCAL_SHELL_OFFSETS = torch.tensor(
    [
        offset
        for offset in product(
            (-LOCAL_SHELL_STEP, 0, LOCAL_SHELL_STEP), repeat=3
        )
        if offset != (0, 0, 0)
    ],
    dtype=torch.long,
)


def local_peak_shape_loss(
    logits: torch.Tensor,
    points: torch.Tensor,
    *,
    margin: float = LOCAL_SHAPE_MARGIN,
) -> torch.Tensor:
    """Rank each annotated center above its valid two-voxel shell.

    Shell locations within 1.5 voxels of any annotated center are ignored.
    This makes the loss invariant to point ordering and prevents one known
    cell from being used as another cell's negative example.
    """

    if logits.ndim != 5 or logits.shape[:2] != (1, 1):
        raise ValueError("logits must have shape (1, 1, Z, Y, X)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    if not len(points):
        return logits.sum() * 0.0

    spatial = torch.tensor(logits.shape[-3:], device=points.device)
    centers = points.round().long()
    valid_centers = ((centers >= 0) & (centers < spatial)).all(dim=1)
    centers = centers[valid_centers]
    valid_points = points[valid_centers]
    if not len(centers):
        return logits.sum() * 0.0

    offsets = LOCAL_SHELL_OFFSETS.to(device=points.device)
    candidates = centers[:, None, :] + offsets[None, :, :]
    in_bounds = ((candidates >= 0) & (candidates < spatial)).all(dim=2)

    # Protect all known centers, including close daughters, from negative use.
    distances = torch.linalg.vector_norm(
        candidates[:, :, None, :].float() - valid_points[None, None, :, :],
        dim=-1,
    )
    protected = (distances <= 1.5).any(dim=2)
    keep = in_bounds & ~protected
    source_indices, shell_indices = torch.nonzero(keep, as_tuple=True)
    if not len(source_indices):
        return logits.sum() * 0.0

    shell = candidates[source_indices, shell_indices]
    center_values = logits[
        0,
        0,
        centers[source_indices, 0],
        centers[source_indices, 1],
        centers[source_indices, 2],
    ]
    shell_values = logits[0, 0, shell[:, 0], shell[:, 1], shell[:, 2]]
    return F.softplus(shell_values - center_values + float(margin)).mean()


def training_loss(
    prediction: dict[str, Any],
    points: torch.Tensor,
    *,
    complete_labels: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if complete_labels:
        loss, components = faint.training_loss(
            prediction, points, complete_labels=True
        )
        return loss, {**components, "local_shape": 0.0}

    logits = prediction["logits"]
    heatmap = expanded.base.positive_logit_loss(logits, points)
    local_shape = local_peak_shape_loss(logits, points)
    offset = expanded.base.subvoxel_offset_loss(prediction["offsets"], [points])
    auxiliary = logits.sum() * 0.0
    for auxiliary_logits in prediction["auxiliary_logits"]:
        scaled = points * (
            auxiliary_logits.shape[-1] / float(logits.shape[-1])
        )
        auxiliary = auxiliary + expanded.base.positive_logit_loss(
            auxiliary_logits, scaled
        )
    rank = logits.sum() * 0.0
    total = (
        heatmap
        + LOCAL_SHAPE_WEIGHT * local_shape
        + 0.20 * offset
        + 0.10 * auxiliary
    )
    return total, {
        "heatmap": float(heatmap.detach()),
        "rank": float(rank.detach()),
        "offset": float(offset.detach()),
        "auxiliary": float(auxiliary.detach()),
        "local_shape": float(local_shape.detach()),
    }


def main() -> None:
    expanded.base.RUN_ID = RUN_ID
    expanded.base.validate_real_manifest = expanded.validate_expanded_real_manifest
    expanded.base.augment_example = faint.augment_example
    expanded.base.training_loss = training_loss
    expanded.base.main()


if __name__ == "__main__":
    main()
