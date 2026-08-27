"""Candidate-limited learned pair evidence for temporal 3D cell association.

This v2 module extends the project-authored physical patch encoder without
changing the v1 cosine model.  It scores only geometrically eligible pairs and
combines appearance persistence, physical displacement, and source-division
evidence in one learned head.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from patch_model import PhysicalPatchAssociationModel
except ModuleNotFoundError:
    from research.temporal_contrastive.patch_model import (
        PhysicalPatchAssociationModel,
    )


PAIR_FUSION_FAMILY = "temporal_pair_fusion_v2"
PAIR_FUSION_POLICY = (
    "candidate-limited source-target-absolute-product-displacement-division MLP"
)
PAIR_LOSS_POLICY = "all-positive candidate-pair mean-log-probability"
PAIR_FEATURE_WIDTH = 1_029
PAIR_PROJECTION_WIDTH = 1_024
PAIR_HIDDEN_WIDTHS = (512, 128)
DEFAULT_CANDIDATE_RADIUS_UM = 32.0
DEFAULT_PAIR_CHUNK_SIZE = 4_096
EXPECTED_PARAMETER_COUNT = 20_869_325


def _positive_radius(value: float) -> float:
    radius = float(value)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("candidate radius must be positive and finite")
    return radius


class PhysicalPairFusionAssociationModel(PhysicalPatchAssociationModel):
    """Physical v1 encoder plus one learned candidate-pair scoring head."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        base_channels: int = 64,
        embedding_channels: int = 256,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            base_channels=base_channels,
            embedding_channels=embedding_channels,
        )
        pair_width = 4 * int(embedding_channels) + 5
        if pair_width != PAIR_FEATURE_WIDTH and embedding_channels == 256:
            raise RuntimeError("declared pair feature width changed")
        self.pair_feature_width = pair_width
        self.pair_head = nn.Sequential(
            nn.LayerNorm(pair_width),
            nn.Linear(pair_width, PAIR_PROJECTION_WIDTH),
            nn.SiLU(inplace=True),
            nn.Linear(PAIR_PROJECTION_WIDTH, PAIR_HIDDEN_WIDTHS[0]),
            nn.SiLU(inplace=True),
            nn.Linear(PAIR_HIDDEN_WIDTHS[0], PAIR_HIDDEN_WIDTHS[1]),
            nn.SiLU(inplace=True),
            nn.Linear(PAIR_HIDDEN_WIDTHS[1], 1),
        )

    def candidate_pair_logits(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        source_coords_zyx_um: torch.Tensor,
        target_coords_zyx_um: torch.Tensor,
        source_division_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
        chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Return a dense logit matrix while evaluating eligible pairs only."""

        if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
            raise ValueError("source and target embeddings must be matrices")
        if source_embeddings.shape[1] != target_embeddings.shape[1]:
            raise ValueError("source and target embedding widths must match")
        source_count, target_count = len(source_embeddings), len(target_embeddings)
        if source_coords_zyx_um.shape != (source_count, 3):
            raise ValueError("source physical coordinates must have shape (N, 3)")
        if target_coords_zyx_um.shape != (target_count, 3):
            raise ValueError("target physical coordinates must have shape (M, 3)")
        if source_division_logits.shape != (source_count,):
            raise ValueError("one source division logit is required per source")
        if candidate_mask.shape != (source_count, target_count):
            raise ValueError("candidate mask does not match the pair matrix")
        if candidate_mask.dtype != torch.bool:
            raise ValueError("candidate mask must be boolean")
        if chunk_size <= 0:
            raise ValueError("pair chunk size must be positive")
        radius = _positive_radius(candidate_radius_um)
        tensors = (
            source_embeddings,
            target_embeddings,
            source_coords_zyx_um,
            target_coords_zyx_um,
            source_division_logits,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("pair-fusion inputs must be finite")
        if not (
            source_embeddings.device
            == target_embeddings.device
            == source_coords_zyx_um.device
            == target_coords_zyx_um.device
            == source_division_logits.device
            == candidate_mask.device
        ):
            raise ValueError("pair-fusion inputs must share one device")

        pair_indices = torch.nonzero(candidate_mask, as_tuple=False)
        output = source_embeddings.new_full(
            (source_count, target_count), float("-inf")
        )
        if len(pair_indices) == 0:
            return output
        source_normalized = F.normalize(source_embeddings, dim=1, eps=1e-8)
        target_normalized = F.normalize(target_embeddings, dim=1, eps=1e-8)
        for start in range(0, len(pair_indices), int(chunk_size)):
            selected = pair_indices[start : start + int(chunk_size)]
            source_rows = selected[:, 0]
            target_rows = selected[:, 1]
            source_values = source_normalized[source_rows]
            target_values = target_normalized[target_rows]
            displacement = (
                target_coords_zyx_um[target_rows]
                - source_coords_zyx_um[source_rows]
            ) / radius
            distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
            division = (
                source_division_logits[source_rows].clamp(-8.0, 8.0) / 8.0
            ).unsqueeze(1)
            features = torch.cat(
                (
                    source_values,
                    target_values,
                    torch.abs(target_values - source_values),
                    source_values * target_values,
                    displacement,
                    distance,
                    division,
                ),
                dim=1,
            )
            if features.shape[1] != self.pair_feature_width:
                raise RuntimeError("pair feature construction changed width")
            logits = self.pair_head(features).squeeze(1)
            output[source_rows, target_rows] = logits
        return output


def masked_multi_positive_pair_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """All-positive row loss for a precomputed candidate-pair logit matrix."""

    if logits.ndim != 2:
        raise ValueError("pair logits must be a matrix")
    if positive_mask.shape != logits.shape or candidate_mask.shape != logits.shape:
        raise ValueError("positive and candidate masks must match pair logits")
    if positive_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
        raise ValueError("positive and candidate masks must be boolean")
    if torch.any(positive_mask & ~candidate_mask):
        raise ValueError("a ground-truth link is absent from the candidate mask")
    if torch.any(torch.isfinite(logits) != candidate_mask):
        raise ValueError("finite pair logits must exactly match candidate eligibility")
    valid = positive_mask.any(dim=1)
    if not torch.any(valid):
        return logits[candidate_mask].sum() * 0.0
    candidates = candidate_mask[valid]
    positives = positive_mask[valid]
    if torch.any(candidates.sum(dim=1) <= positives.sum(dim=1)):
        raise ValueError("each labeled source needs at least one hard negative")
    valid_logits = logits[valid]
    all_logsumexp = torch.logsumexp(valid_logits, dim=1)
    positive_mean_logit = (
        (valid_logits.masked_fill(~positives, 0.0) * positives).sum(dim=1)
        / positives.sum(dim=1).to(valid_logits.dtype)
    )
    return (all_logsumexp - positive_mean_logit).mean()


def pair_logit_metrics(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, float | int]:
    """Rank true children using learned pair logits instead of cosine."""

    if logits.shape != candidate_mask.shape or logits.shape != positive_mask.shape:
        raise ValueError("pair metric matrices must have equal shape")
    if candidate_mask.dtype != torch.bool or positive_mask.dtype != torch.bool:
        raise ValueError("pair metric masks must be boolean")
    if torch.any(positive_mask & ~candidate_mask):
        raise ValueError("pair metrics omitted a positive")
    if torch.any(torch.isfinite(logits) != candidate_mask):
        raise ValueError("pair metric logits do not match candidates")
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    top1 = positive_mask.gather(1, order[:, :1]).any(dim=1).float()
    ranked_positive = positive_mask.gather(1, order)
    ranks = torch.argmax(ranked_positive.to(torch.int64), dim=1) + 1
    divisions = positive_mask.sum(dim=1) >= 2
    if torch.any(divisions):
        top2_division = (
            positive_mask[divisions]
            .gather(1, order[divisions, :2])
            .sum(dim=1)
            >= 2
        ).float()
        division_recall = float(top2_division.mean())
        division_rows = int(divisions.sum())
    else:
        division_recall = 0.0
        division_rows = 0
    return {
        "top1": float(top1.mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "division_top2": division_recall,
        "rows": len(logits),
        "division_rows": division_rows,
    }


@torch.inference_mode()
def pair_fusion_scores_for_movie(
    model: PhysicalPairFusionAssociationModel,
    video: Any,
    node_embeddings: np.ndarray,
    node_division_logits: np.ndarray,
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    voxel_size_zyx_um: Sequence[float] = (1.625, 0.40625, 0.40625),
    candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
    chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
) -> dict[int, np.ndarray]:
    """Score only finite Trackastra candidates and return neutral elsewhere."""

    embeddings = np.asarray(node_embeddings, dtype=np.float32)
    divisions = np.asarray(node_division_logits, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(video.node_ids):
        raise ValueError("node embeddings do not align with the video")
    if divisions.shape != (len(video.node_ids),):
        raise ValueError("node division logits do not align with the video")
    spacing = np.asarray(tuple(voxel_size_zyx_um), dtype=np.float32)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("voxel size must contain three positive finite values")
    coords_um = np.asarray(video.coords_voxel, dtype=np.float32) * spacing[None]
    if coords_um.shape != (len(video.node_ids), 3) or not np.isfinite(coords_um).all():
        raise ValueError("video coordinates do not align with node identifiers")
    id_to_row = {
        int(node_id): row for row, node_id in enumerate(video.node_ids.tolist())
    }
    device = next(model.parameters()).device
    embedding_tensor = torch.as_tensor(embeddings, device=device)
    division_tensor = torch.as_tensor(divisions, device=device)
    coordinate_tensor = torch.as_tensor(coords_um, device=device)
    result: dict[int, np.ndarray] = {}
    for timepoint, (source_ids, target_ids, track_scores) in pair_scores.items():
        source_rows = np.asarray([id_to_row[int(node)] for node in source_ids])
        target_rows = np.asarray([id_to_row[int(node)] for node in target_ids])
        candidates = np.isfinite(np.asarray(track_scores))
        logits = model.candidate_pair_logits(
            embedding_tensor[source_rows],
            embedding_tensor[target_rows],
            coordinate_tensor[source_rows],
            coordinate_tensor[target_rows],
            division_tensor[source_rows],
            torch.as_tensor(candidates, dtype=torch.bool, device=device),
            candidate_radius_um=candidate_radius_um,
            chunk_size=chunk_size,
        )
        probabilities = np.full(candidates.shape, 0.5, dtype=np.float32)
        if np.any(candidates):
            finite_probabilities = torch.sigmoid(logits[torch.isfinite(logits)])
            probabilities[candidates] = finite_probabilities.float().cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("pair-fusion model produced non-finite evidence")
        result[int(timepoint)] = probabilities
    return result
