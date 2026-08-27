"""Biohub transition adapter for the staged contextual pair-fusion v3 model."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

try:
    import train_dual_fold_patch as base
    from contextual_pair_fusion import (
        DEFAULT_PAIR_CHUNK_SIZE,
        ContextualPairFusionAssociationModel,
    )
    from pair_fusion import DEFAULT_CANDIDATE_RADIUS_UM, pair_logit_metrics
    from transition_context import (
        TransitionContext,
        candidate_transition_features,
        estimate_transition_context,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive import train_dual_fold_patch as base
    from research.temporal_contrastive.contextual_pair_fusion import (
        DEFAULT_PAIR_CHUNK_SIZE,
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.pair_fusion import (
        DEFAULT_CANDIDATE_RADIUS_UM,
        pair_logit_metrics,
    )
    from research.temporal_contrastive.transition_context import (
        TransitionContext,
        candidate_transition_features,
        estimate_transition_context,
    )


def physical_coordinates(
    values: np.ndarray,
    voxel_size_zyx_um: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    coords = torch.as_tensor(values, dtype=torch.float32, device=device)
    spacing = torch.as_tensor(
        tuple(voxel_size_zyx_um), dtype=torch.float32, device=device
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("node coordinates must have shape (N, 3)")
    if spacing.shape != (3,) or not torch.isfinite(spacing).all() or torch.any(
        spacing <= 0
    ):
        raise ValueError("voxel size must contain three positive finite values")
    return coords * spacing[None]


def image_transition_context(
    source_volume: np.ndarray,
    target_volume: np.ndarray,
    *,
    voxel_size_zyx_um: Sequence[float],
) -> TransitionContext:
    """Use the central frames from overlapping ``t-1,t,t+1`` contexts."""

    source = np.asarray(source_volume, dtype=np.float32)
    target = np.asarray(target_volume, dtype=np.float32)
    if source.ndim != 4 or target.ndim != 4 or source.shape[0] != 3 or target.shape[0] != 3:
        raise ValueError("temporal volumes must have shape (3, Z, Y, X)")
    if source.shape != target.shape:
        raise ValueError("source and target temporal volumes must have equal shape")
    return estimate_transition_context(
        source[1], target[1], voxel_size_zyx_um=voxel_size_zyx_um
    )


def contextual_pair_logits_for_transition(
    model: ContextualPairFusionAssociationModel,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    source_division_logits: torch.Tensor,
    batch: base.TransitionBatch,
    source_volume: np.ndarray,
    target_volume: np.ndarray,
    voxel_size_zyx_um: Sequence[float],
    device: torch.device,
    *,
    candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
    pair_chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TransitionContext]:
    """Build image/motion context and score one complete candidate transition."""

    candidates = torch.as_tensor(
        batch.candidate_mask, dtype=torch.bool, device=device
    )
    positives = torch.as_tensor(
        batch.positive_mask, dtype=torch.bool, device=device
    )
    transition_context = image_transition_context(
        source_volume,
        target_volume,
        voxel_size_zyx_um=voxel_size_zyx_um,
    )
    source_coords_um = physical_coordinates(
        batch.source_coords, voxel_size_zyx_um, device
    )
    target_coords_um = physical_coordinates(
        batch.target_coords, voxel_size_zyx_um, device
    )
    candidate_context = candidate_transition_features(
        source_coords_um.detach().cpu().numpy(),
        target_coords_um.detach().cpu().numpy(),
        np.asarray(batch.candidate_mask),
        transition_context,
        candidate_radius_um=candidate_radius_um,
    )
    logits = model.candidate_pair_logits(
        source_embeddings,
        target_embeddings,
        source_coords_um,
        target_coords_um,
        source_division_logits,
        candidates,
        torch.as_tensor(candidate_context, dtype=torch.float32, device=device),
        candidate_radius_um=candidate_radius_um,
        chunk_size=pair_chunk_size,
    )
    return logits, candidates, positives, transition_context


@torch.no_grad()
def validate_contextual_model(
    model: ContextualPairFusionAssociationModel,
    examples: list[
        tuple[
            np.ndarray,
            np.ndarray,
            base.TransitionBatch,
            tuple[float, float, float],
        ]
    ],
    device: torch.device,
    *,
    candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
    pair_chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
) -> dict[str, float | int]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    for source_volume, target_volume, batch, voxel_size in examples:
        source, target, divisions = base.encode_transition(
            model,
            source_volume,
            target_volume,
            batch,
            voxel_size,
            device,
            augment=False,
        )
        logits, candidates, positives, _context = (
            contextual_pair_logits_for_transition(
                model,
                source,
                target,
                divisions,
                batch,
                source_volume,
                target_volume,
                voxel_size,
                device,
                candidate_radius_um=candidate_radius_um,
                pair_chunk_size=pair_chunk_size,
            )
        )
        rows.append(pair_logit_metrics(logits, candidates, positives))
    model.train()
    return base.aggregate_metrics(rows)
