"""Biohub-owned temporal embedding and division heads.

This module implements the transferable modeling idea identified in CELLECT's
paper without copying its architecture or source. It is designed to sit on top
of high-resolution features from the independent SpatialDINO detector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemporalFusionHead(nn.Module):
    """Fuse adjacent-frame feature maps into link embeddings and division logits."""

    def __init__(
        self,
        feature_channels: int,
        *,
        hidden_channels: int = 128,
        embedding_channels: int = 64,
    ) -> None:
        super().__init__()
        if min(feature_channels, hidden_channels, embedding_channels) <= 0:
            raise ValueError("all channel counts must be positive")
        groups = next(
            group for group in (8, 4, 2, 1) if hidden_channels % group == 0
        )
        # Concatenating the two directions, their change, and their product
        # exposes motion and persistence while retaining full spatial support.
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_channels * 4, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.embedding = nn.Conv3d(
            hidden_channels, embedding_channels, kernel_size=1
        )
        self.division = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        nn.init.constant_(self.division.bias, -4.0)

    def forward(
        self, current: torch.Tensor, following: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current.shape != following.shape or current.ndim != 5:
            raise ValueError("adjacent features must have the same 5D shape")
        values = torch.cat(
            (current, following, following - current, following * current), dim=1
        )
        fused = self.fusion(values)
        embeddings = F.normalize(self.embedding(fused), p=2, dim=1, eps=1e-8)
        return embeddings, self.division(fused)


def masked_link_info_nce(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    positive_target_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Contrast true links only against geometrically feasible hard negatives.

    Rows whose positive target is ``-1`` are unlabeled and excluded. Every
    labeled positive must also be present in ``candidate_mask``; silently adding
    a missing positive would hide a candidate-coverage error.
    """

    if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("source and target embeddings must be matrices")
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("source and target embedding widths must match")
    if positive_target_indices.shape != (source_embeddings.shape[0],):
        raise ValueError("one positive target index is required per source")
    if candidate_mask.shape != (
        source_embeddings.shape[0],
        target_embeddings.shape[0],
    ):
        raise ValueError("candidate_mask has the wrong shape")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    valid = positive_target_indices >= 0
    if not torch.any(valid):
        return source_embeddings.sum() * 0.0
    positive = positive_target_indices[valid].long()
    if torch.any(positive >= target_embeddings.shape[0]):
        raise ValueError("a positive target index is outside the target matrix")
    valid_candidates = candidate_mask[valid]
    if not torch.all(valid_candidates.gather(1, positive[:, None]).squeeze(1)):
        raise ValueError("a ground-truth link is absent from the candidate mask")
    if torch.any(valid_candidates.sum(dim=1) < 2):
        raise ValueError("each labeled source needs at least one hard negative")

    source = F.normalize(source_embeddings[valid], p=2, dim=1, eps=1e-8)
    target = F.normalize(target_embeddings, p=2, dim=1, eps=1e-8)
    logits = (source @ target.transpose(0, 1)) / float(temperature)
    logits = logits.masked_fill(~valid_candidates, torch.finfo(logits.dtype).min)
    return F.cross_entropy(logits, positive)


def masked_multi_positive_info_nce(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Contrast one or more true children against feasible target cells.

    Unlike a single-label cross entropy, this objective represents divisions
    without declaring one true daughter a negative. Sources without a labeled
    child are excluded because the organizer graphs are sparse annotations, not
    proof that an unlinked detection truly disappears.
    """

    if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("source and target embeddings must be matrices")
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("source and target embedding widths must match")
    expected = (source_embeddings.shape[0], target_embeddings.shape[0])
    if positive_mask.shape != expected or candidate_mask.shape != expected:
        raise ValueError("positive and candidate masks must match the pair matrix")
    if positive_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
        raise ValueError("positive and candidate masks must be boolean")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if torch.any(positive_mask & ~candidate_mask):
        raise ValueError("a ground-truth link is absent from the candidate mask")

    valid = positive_mask.any(dim=1)
    if not torch.any(valid):
        return source_embeddings.sum() * 0.0
    candidates = candidate_mask[valid]
    if torch.any(candidates.sum(dim=1) <= positive_mask[valid].sum(dim=1)):
        raise ValueError("each labeled source needs at least one hard negative")

    source = F.normalize(source_embeddings[valid], p=2, dim=1, eps=1e-8)
    target = F.normalize(target_embeddings, p=2, dim=1, eps=1e-8)
    logits = (source @ target.transpose(0, 1)) / float(temperature)
    floor = torch.finfo(logits.dtype).min
    all_logsumexp = torch.logsumexp(logits.masked_fill(~candidates, floor), dim=1)
    positive_logsumexp = torch.logsumexp(
        logits.masked_fill(~positive_mask[valid], floor), dim=1
    )
    return (all_logsumexp - positive_logsumexp).mean()
