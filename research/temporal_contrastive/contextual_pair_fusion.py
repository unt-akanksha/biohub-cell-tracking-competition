"""Project-authored candidate-set context head for temporal Biohub v3.

The model keeps the physical temporal 3D encoder used by the independent v1
control.  Each eligible edge receives image-derived transition context, then
compares its learned token with the mean and maximum tokens of competing
outgoing and incoming edges.  No public implementation, weights, or predictions
are used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from pair_fusion import DEFAULT_CANDIDATE_RADIUS_UM
    from patch_model import PhysicalPatchAssociationModel
    from transition_context import CANDIDATE_CONTEXT_WIDTH
except ModuleNotFoundError:
    from research.temporal_contrastive.pair_fusion import (
        DEFAULT_CANDIDATE_RADIUS_UM,
    )
    from research.temporal_contrastive.patch_model import (
        PhysicalPatchAssociationModel,
    )
    from research.temporal_contrastive.transition_context import (
        CANDIDATE_CONTEXT_WIDTH,
    )


CONTEXTUAL_PAIR_FUSION_FAMILY = "temporal_contextual_pair_fusion_v3"
CONTEXTUAL_PAIR_POLICY = (
    "candidate-limited temporal-context outgoing-incoming edge-set pooling"
)
BASE_PAIR_FEATURE_WIDTH = 1_029
CONTEXTUAL_PAIR_FEATURE_WIDTH = BASE_PAIR_FEATURE_WIDTH + CANDIDATE_CONTEXT_WIDTH
EDGE_TOKEN_HIDDEN_WIDTH = 512
EDGE_TOKEN_WIDTH = 256
EDGE_SET_FEATURE_WIDTH = 6 * EDGE_TOKEN_WIDTH
EDGE_HEAD_HIDDEN_WIDTHS = (512, 128)
DEFAULT_PAIR_CHUNK_SIZE = 4_096
EXPECTED_PARAMETER_COUNT = 20_747_761


def _positive_radius(value: float) -> float:
    radius = float(value)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("candidate radius must be positive and finite")
    return radius


def _group_mean_and_max(
    tokens: torch.Tensor, group_rows: torch.Tensor, group_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2 or group_rows.shape != (len(tokens),):
        raise ValueError("edge tokens and group rows do not align")
    if group_rows.dtype != torch.long:
        raise ValueError("group rows must be int64")
    if group_count <= 0 or torch.any(group_rows < 0) or torch.any(group_rows >= group_count):
        raise ValueError("edge group row is outside the declared inventory")
    width = tokens.shape[1]
    expanded_rows = group_rows[:, None].expand(-1, width)
    totals = tokens.new_zeros((group_count, width))
    totals.scatter_add_(0, expanded_rows, tokens)
    counts = tokens.new_zeros((group_count, 1))
    counts.scatter_add_(0, group_rows[:, None], tokens.new_ones((len(tokens), 1)))
    means = totals / counts.clamp_min(1.0)
    maxima = tokens.new_full((group_count, width), float("-inf"))
    maxima.scatter_reduce_(
        0, expanded_rows, tokens, reduce="amax", include_self=True
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return means, maxima


class ContextualPairFusionAssociationModel(PhysicalPatchAssociationModel):
    """Temporal encoder plus learned outgoing/incoming candidate-set context."""

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
        base_pair_width = 4 * int(embedding_channels) + 5
        if base_pair_width != BASE_PAIR_FEATURE_WIDTH and embedding_channels == 256:
            raise RuntimeError("declared v3 base pair width changed")
        self.pair_feature_width = base_pair_width + CANDIDATE_CONTEXT_WIDTH
        self.edge_token = nn.Sequential(
            nn.LayerNorm(self.pair_feature_width),
            nn.Linear(self.pair_feature_width, EDGE_TOKEN_HIDDEN_WIDTH),
            nn.SiLU(inplace=True),
            nn.Linear(EDGE_TOKEN_HIDDEN_WIDTH, EDGE_TOKEN_WIDTH),
            nn.SiLU(inplace=True),
        )
        self.edge_head = nn.Sequential(
            nn.LayerNorm(EDGE_SET_FEATURE_WIDTH),
            nn.Linear(EDGE_SET_FEATURE_WIDTH, EDGE_HEAD_HIDDEN_WIDTHS[0]),
            nn.SiLU(inplace=True),
            nn.Linear(EDGE_HEAD_HIDDEN_WIDTHS[0], EDGE_HEAD_HIDDEN_WIDTHS[1]),
            nn.SiLU(inplace=True),
            nn.Linear(EDGE_HEAD_HIDDEN_WIDTHS[1], 1),
        )

    def candidate_pair_logits(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        source_coords_zyx_um: torch.Tensor,
        target_coords_zyx_um: torch.Tensor,
        source_division_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_context: torch.Tensor,
        *,
        candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
        chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Score eligible edges after outgoing/incoming set-context pooling."""

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
        if candidate_context.shape != (
            source_count,
            target_count,
            CANDIDATE_CONTEXT_WIDTH,
        ):
            raise ValueError("candidate context has the wrong dense shape")
        if chunk_size <= 0:
            raise ValueError("pair chunk size must be positive")
        radius = _positive_radius(candidate_radius_um)
        tensors = (
            source_embeddings,
            target_embeddings,
            source_coords_zyx_um,
            target_coords_zyx_um,
            source_division_logits,
            candidate_context,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("contextual pair inputs must be finite")
        if not all(value.device == source_embeddings.device for value in tensors[1:]):
            raise ValueError("contextual pair inputs must share one device")
        if candidate_mask.device != source_embeddings.device:
            raise ValueError("candidate mask must share the model input device")

        pair_indices = torch.nonzero(candidate_mask, as_tuple=False)
        output = source_embeddings.new_full(
            (source_count, target_count), float("-inf")
        )
        if not len(pair_indices):
            return output
        source_normalized = F.normalize(source_embeddings, dim=1, eps=1e-8)
        target_normalized = F.normalize(target_embeddings, dim=1, eps=1e-8)
        token_chunks: list[torch.Tensor] = []
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
            raw_features = torch.cat(
                (
                    source_values,
                    target_values,
                    torch.abs(target_values - source_values),
                    source_values * target_values,
                    displacement,
                    distance,
                    division,
                    candidate_context[source_rows, target_rows],
                ),
                dim=1,
            )
            if raw_features.shape[1] != self.pair_feature_width:
                raise RuntimeError("contextual pair feature construction changed")
            token_chunks.append(self.edge_token(raw_features))
        tokens = torch.cat(token_chunks, dim=0)
        source_rows = pair_indices[:, 0]
        target_rows = pair_indices[:, 1]
        source_mean, source_max = _group_mean_and_max(
            tokens, source_rows, source_count
        )
        target_mean, target_max = _group_mean_and_max(
            tokens, target_rows, target_count
        )
        transition_mean = tokens.mean(dim=0, keepdim=True).expand(len(tokens), -1)
        edge_set_features = torch.cat(
            (
                tokens,
                source_mean[source_rows],
                source_max[source_rows],
                target_mean[target_rows],
                target_max[target_rows],
                transition_mean,
            ),
            dim=1,
        )
        if edge_set_features.shape[1] != EDGE_SET_FEATURE_WIDTH:
            raise RuntimeError("edge-set context width changed")
        logits = self.edge_head(edge_set_features).squeeze(1)
        output[source_rows, target_rows] = logits
        return output
