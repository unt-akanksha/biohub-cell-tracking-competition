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
    from pair_fusion import (
        DEFAULT_CANDIDATE_RADIUS_UM,
        masked_multi_positive_pair_nll,
    )
    from patch_model import PhysicalPatchAssociationModel
    from transition_context import (
        CANDIDATE_CONTEXT_WIDTH,
        candidate_transition_features,
        estimate_transition_context,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.pair_fusion import (
        DEFAULT_CANDIDATE_RADIUS_UM,
        masked_multi_positive_pair_nll,
    )
    from research.temporal_contrastive.patch_model import (
        PhysicalPatchAssociationModel,
    )
    from research.temporal_contrastive.transition_context import (
        CANDIDATE_CONTEXT_WIDTH,
        candidate_transition_features,
        estimate_transition_context,
    )


CONTEXTUAL_PAIR_FUSION_FAMILY = "temporal_contextual_pair_fusion_v3"
CONTEXTUAL_PAIR_POLICY = (
    "candidate-limited temporal-context outgoing-incoming edge-set pooling"
)
TRANSITION_CONTEXT_POLICY = (
    "bounded phase-correlation with projection refinement, duplicate evidence, "
    "and robust residual-motion context"
)
CONTEXTUAL_PAIR_LOSS_POLICY = (
    "outgoing all-positive child ranking plus eligible incoming one-parent ranking"
)
RECIPROCAL_PARENT_LOSS_WEIGHT = 0.35
BASE_PAIR_FEATURE_WIDTH = 1_029
CONTEXTUAL_PAIR_FEATURE_WIDTH = BASE_PAIR_FEATURE_WIDTH + CANDIDATE_CONTEXT_WIDTH
EDGE_TOKEN_HIDDEN_WIDTH = 512
EDGE_TOKEN_WIDTH = 256
EDGE_SET_FEATURE_WIDTH = 6 * EDGE_TOKEN_WIDTH
EDGE_HEAD_HIDDEN_WIDTHS = (512, 128)
DEFAULT_PAIR_CHUNK_SIZE = 4_096
EXPECTED_PARAMETER_COUNT = 20_747_761


def masked_reciprocal_parent_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Rank each labeled child among candidate parents when competition exists.

    A target with only its true parent in the candidate set provides no
    incoming ranking signal and is skipped. This preserves the sparse-label
    contract while teaching the biological at-most-one-parent constraint.
    """

    if logits.ndim != 2:
        raise ValueError("contextual pair logits must be a matrix")
    if positive_mask.shape != logits.shape or candidate_mask.shape != logits.shape:
        raise ValueError("positive and candidate masks must match pair logits")
    if positive_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
        raise ValueError("positive and candidate masks must be boolean")
    if torch.any(positive_mask & ~candidate_mask):
        raise ValueError("a ground-truth link is absent from the candidate mask")
    if torch.any(torch.isfinite(logits) != candidate_mask):
        raise ValueError("finite pair logits must exactly match candidate eligibility")
    incoming_logits = logits.transpose(0, 1)
    incoming_positives = positive_mask.transpose(0, 1)
    incoming_candidates = candidate_mask.transpose(0, 1)
    eligible = incoming_positives.any(dim=1) & (
        incoming_candidates.sum(dim=1) > incoming_positives.sum(dim=1)
    )
    if not torch.any(eligible):
        return logits[candidate_mask].sum() * 0.0
    return masked_multi_positive_pair_nll(
        incoming_logits[eligible],
        incoming_positives[eligible],
        incoming_candidates[eligible],
    )


def contextual_bidirectional_pair_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Combine outgoing lineage ranking with an incoming-parent constraint."""

    outgoing = masked_multi_positive_pair_nll(
        logits, positive_mask, candidate_mask
    )
    incoming = masked_reciprocal_parent_nll(
        logits, positive_mask, candidate_mask
    )
    return outgoing + RECIPROCAL_PARENT_LOSS_WEIGHT * incoming


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
        # Autocast keeps the dense sentinel matrix in the input embedding dtype
        # while the learned edge head emits fp16/bfloat16 values.  Index-put
        # requires an exact dtype match, so promote the compact learned logits
        # back to the stable output dtype before scattering them.  The cast is
        # differentiable and keeps losses/softmaxes in fp32 during training.
        output[source_rows, target_rows] = logits.to(dtype=output.dtype)
        return output


@torch.inference_mode()
def contextual_pair_fusion_scores_for_movie(
    model: ContextualPairFusionAssociationModel,
    video: object,
    image: object,
    node_embeddings: np.ndarray,
    node_division_logits: np.ndarray,
    pair_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    voxel_size_zyx_um: tuple[float, float, float] = (
        1.625,
        0.40625,
        0.40625,
    ),
    candidate_radius_um: float = DEFAULT_CANDIDATE_RADIUS_UM,
    chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
) -> dict[int, np.ndarray]:
    """Score whole-movie candidates with the same v3 context used in training."""

    node_ids = np.asarray(getattr(video, "node_ids"), dtype=np.int64)
    coords_voxel = np.asarray(getattr(video, "coords_voxel"), dtype=np.float32)
    embeddings = np.asarray(node_embeddings, dtype=np.float32)
    divisions = np.asarray(node_division_logits, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(node_ids):
        raise ValueError("node embeddings do not align with the video")
    if divisions.shape != (len(node_ids),):
        raise ValueError("node division logits do not align with the video")
    spacing = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("voxel size must contain three positive finite values")
    coords_um = coords_voxel * spacing[None]
    if coords_um.shape != (len(node_ids), 3) or not np.isfinite(coords_um).all():
        raise ValueError("video coordinates do not align with node identifiers")
    image_shape = tuple(int(value) for value in getattr(image, "shape"))
    if len(image_shape) != 4 or image_shape[0] < 2:
        raise ValueError("movie image must have shape (T, Z, Y, X)")
    id_to_row = {int(node_id): row for row, node_id in enumerate(node_ids)}
    device = next(model.parameters()).device
    embedding_tensor = torch.as_tensor(embeddings, device=device)
    division_tensor = torch.as_tensor(divisions, device=device)
    coordinate_tensor = torch.as_tensor(coords_um, device=device)
    frame_cache: dict[int, np.ndarray] = {}

    def frame(timepoint: int) -> np.ndarray:
        if timepoint not in frame_cache:
            value = np.asarray(image[timepoint], dtype=np.float32)
            if value.shape != image_shape[1:] or not np.isfinite(value).all():
                raise ValueError("movie frame shape or values changed")
            frame_cache[timepoint] = value
        return frame_cache[timepoint]

    result: dict[int, np.ndarray] = {}
    for timepoint, (source_ids, target_ids, track_scores) in sorted(
        pair_scores.items()
    ):
        current_time = int(timepoint)
        if not 0 <= current_time < image_shape[0] - 1:
            raise ValueError("candidate transition timepoint is outside the movie")
        source_rows = np.asarray(
            [id_to_row[int(node)] for node in source_ids], dtype=np.int64
        )
        target_rows = np.asarray(
            [id_to_row[int(node)] for node in target_ids], dtype=np.int64
        )
        candidates = np.isfinite(np.asarray(track_scores))
        transition = estimate_transition_context(
            frame(current_time),
            frame(current_time + 1),
            voxel_size_zyx_um=spacing,
        )
        context = candidate_transition_features(
            coords_um[source_rows],
            coords_um[target_rows],
            candidates,
            transition,
            candidate_radius_um=candidate_radius_um,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model.candidate_pair_logits(
                embedding_tensor[source_rows],
                embedding_tensor[target_rows],
                coordinate_tensor[source_rows],
                coordinate_tensor[target_rows],
                division_tensor[source_rows],
                torch.as_tensor(candidates, dtype=torch.bool, device=device),
                torch.as_tensor(context, dtype=torch.float32, device=device),
                candidate_radius_um=candidate_radius_um,
                chunk_size=chunk_size,
            )
        probabilities = np.full(candidates.shape, 0.5, dtype=np.float32)
        if np.any(candidates):
            probabilities[candidates] = (
                torch.sigmoid(logits[torch.isfinite(logits)]).float().cpu().numpy()
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("contextual pair model produced non-finite evidence")
        result[current_time] = probabilities
        stale = [key for key in frame_cache if key < current_time]
        for key in stale:
            del frame_cache[key]
    return result
