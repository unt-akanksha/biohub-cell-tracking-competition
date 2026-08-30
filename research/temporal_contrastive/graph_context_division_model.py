"""Heavy image-plus-detection-context model for division recovery ranking."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

try:
    from graph_context_archive_contract import (
        CONTEXT_FEATURE_WIDTH,
        CONTEXT_TOKEN_COUNT,
    )
except ModuleNotFoundError:
    CONTEXT_TOKEN_COUNT = 43
    CONTEXT_FEATURE_WIDTH = 8

try:
    from relational_division_model import (
        BACKBONE_LOGIT_WIDTH,
        BACKBONE_PARAMETER_COUNT,
        CENTER_COUNT,
        NORMALIZED_GEOMETRY_WIDTH,
        RELATIONAL_VISUAL_BLOCKS,
        TEMPORAL_CHANNELS,
        normalize_geometry,
    )
    from multiscale_contextual_pair_fusion import (
        MultiscaleContextualPairFusionAssociationModel,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.relational_division_model import (
        BACKBONE_LOGIT_WIDTH,
        BACKBONE_PARAMETER_COUNT,
        CENTER_COUNT,
        NORMALIZED_GEOMETRY_WIDTH,
        RELATIONAL_VISUAL_BLOCKS,
        TEMPORAL_CHANNELS,
        normalize_geometry,
    )
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        MultiscaleContextualPairFusionAssociationModel,
    )


GRAPH_CONTEXT_DIVISION_FAMILY = "temporal_multiscale_graph_context_division_v1"
DEFAULT_EMBEDDING_CHANNELS = 256
DEFAULT_CONTEXT_DIMENSION = 512
DEFAULT_CONTEXT_HEADS = 8
DEFAULT_CONTEXT_LAYERS = 8
DEFAULT_CONTEXT_FEEDFORWARD = 2_048
DEFAULT_HEAD_WIDTHS = (1_024, 512, 128)


class GraphContextDivisionModel(nn.Module):
    """Shared microscopy CNN plus a permutation-invariant detection transformer."""

    def __init__(
        self,
        *,
        base_channels: int = 64,
        embedding_channels: int = DEFAULT_EMBEDDING_CHANNELS,
        projection_base_channels: int = 96,
        context_dimension: int = DEFAULT_CONTEXT_DIMENSION,
        context_heads: int = DEFAULT_CONTEXT_HEADS,
        context_layers: int = DEFAULT_CONTEXT_LAYERS,
        context_feedforward: int = DEFAULT_CONTEXT_FEEDFORWARD,
        dropout: float = 0.10,
        head_widths: tuple[int, ...] = DEFAULT_HEAD_WIDTHS,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("graph-context dropout must be in [0, 1)")
        if min(context_dimension, context_heads, context_layers, context_feedforward) <= 0:
            raise ValueError("graph-context transformer dimensions must be positive")
        if context_dimension % context_heads:
            raise ValueError("graph-context dimension must be divisible by head count")
        if not head_widths or min(head_widths) <= 0:
            raise ValueError("graph-context head widths must be positive")

        self.embedding_channels = int(embedding_channels)
        self.context_dimension = int(context_dimension)
        self.backbone = MultiscaleContextualPairFusionAssociationModel(
            base_channels=base_channels,
            embedding_channels=embedding_channels,
            projection_base_channels=projection_base_channels,
        )
        relational_width = (
            RELATIONAL_VISUAL_BLOCKS * self.embedding_channels
            + BACKBONE_LOGIT_WIDTH
            + NORMALIZED_GEOMETRY_WIDTH
        )
        self.relational_feature_width = relational_width
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(relational_width),
            nn.Linear(relational_width, context_dimension),
            nn.GELU(),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(CONTEXT_FEATURE_WIDTH),
            nn.Linear(CONTEXT_FEATURE_WIDTH, context_dimension),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_dimension,
            nhead=context_heads,
            dim_feedforward=context_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=context_layers,
            norm=nn.LayerNorm(context_dimension),
            enable_nested_tensor=False,
        )
        head_input = relational_width + context_dimension
        layers: list[nn.Module] = [nn.LayerNorm(head_input)]
        input_width = head_input
        for output_width in head_widths:
            layers.extend(
                (
                    nn.Linear(input_width, output_width),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            input_width = output_width
        layers.append(nn.Linear(input_width, 1))
        self.ranking_head = nn.Sequential(*layers)

    def relational_features(
        self, patches: torch.Tensor, geometry: torch.Tensor
    ) -> torch.Tensor:
        if patches.ndim != 6 or tuple(patches.shape[1:3]) != (
            CENTER_COUNT,
            TEMPORAL_CHANNELS,
        ):
            raise ValueError(
                "graph-context patches must have shape (N, 3 centers, 3 temporal, Z, Y, X)"
            )
        if geometry.shape != (len(patches), 9):
            raise ValueError(f"geometry must have shape ({len(patches)}, 9)")
        flattened = patches.reshape(-1, *patches.shape[2:])
        embeddings, division_logits = self.backbone(flattened)
        embeddings = embeddings.reshape(len(patches), CENTER_COUNT, self.embedding_channels)
        division_logits = division_logits.reshape(len(patches), CENTER_COUNT)
        parent = embeddings[:, 0]
        daughters = embeddings[:, 1:]
        daughter_mean = daughters.mean(dim=1)
        daughter_difference = torch.abs(daughters[:, 0] - daughters[:, 1])
        visual = torch.cat(
            (
                parent,
                daughter_mean,
                daughter_difference,
                parent * daughter_mean,
                torch.abs(parent - daughter_mean),
            ),
            dim=1,
        )
        daughter_logits = division_logits[:, 1:]
        logit_features = torch.stack(
            (
                division_logits[:, 0],
                daughter_logits.mean(dim=1),
                torch.abs(daughter_logits[:, 0] - daughter_logits[:, 1]),
            ),
            dim=1,
        )
        features = torch.cat((visual, logit_features, normalize_geometry(geometry)), dim=1)
        if features.shape[1] != self.relational_feature_width:
            raise RuntimeError("graph-context relational feature width changed")
        return features

    def forward(
        self,
        patches: torch.Tensor,
        geometry: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.shape != (
            len(patches),
            CONTEXT_TOKEN_COUNT,
            CONTEXT_FEATURE_WIDTH,
        ):
            raise ValueError(
                "context must have shape "
                f"({len(patches)}, {CONTEXT_TOKEN_COUNT}, {CONTEXT_FEATURE_WIDTH})"
            )
        if context_mask.shape != (len(patches), CONTEXT_TOKEN_COUNT):
            raise ValueError(
                f"context mask must have shape ({len(patches)}, {CONTEXT_TOKEN_COUNT})"
            )
        valid = context_mask.to(dtype=torch.bool)
        parent_anchors = ((context[..., 5] > 0.5) & valid).sum(dim=1)
        daughter_anchors = ((context[..., 6] > 0.5) & valid).sum(dim=1)
        if (
            torch.any(valid.sum(dim=1) < 3)
            or torch.any(parent_anchors != 1)
            or torch.any(daughter_anchors != 2)
        ):
            raise ValueError("graph-context anchor tokens must always be valid")
        relational = self.relational_features(patches, geometry)
        candidate = self.candidate_projection(relational).unsqueeze(1)
        context_tokens = self.context_projection(context.float())
        tokens = torch.cat((candidate, context_tokens), dim=1)
        candidate_mask = torch.ones((len(patches), 1), dtype=torch.bool, device=valid.device)
        token_mask = torch.cat((candidate_mask, valid), dim=1)
        encoded = self.context_encoder(
            tokens,
            src_key_padding_mask=~token_mask,
        )
        fused = torch.cat((relational, encoded[:, 0]), dim=1)
        return self.ranking_head(fused).squeeze(1)


def load_backbone_checkpoint(
    model: GraphContextDivisionModel, state_dict: Mapping[str, torch.Tensor]
) -> None:
    expected = model.backbone.state_dict()
    if set(state_dict) != set(expected):
        raise ValueError("graph-context warm-start backbone keys changed")
    mismatched = [
        name
        for name, value in state_dict.items()
        if tuple(value.shape) != tuple(expected[name].shape)
    ]
    if mismatched:
        raise ValueError(f"graph-context warm-start shapes changed: {mismatched}")
    model.backbone.load_state_dict(dict(state_dict), strict=True)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def architecture_contract() -> dict[str, object]:
    model = GraphContextDivisionModel()
    backbone_count = parameter_count(model.backbone)
    if backbone_count != BACKBONE_PARAMETER_COUNT:
        raise RuntimeError("graph-context backbone inventory changed")
    total = parameter_count(model)
    return {
        "family": GRAPH_CONTEXT_DIVISION_FAMILY,
        "parameter_count": total,
        "backbone_parameter_count": backbone_count,
        "graph_context_parameter_count": total - backbone_count,
        "context_token_count": CONTEXT_TOKEN_COUNT,
        "context_feature_width": CONTEXT_FEATURE_WIDTH,
        "context_dimension": DEFAULT_CONTEXT_DIMENSION,
        "context_heads": DEFAULT_CONTEXT_HEADS,
        "context_layers": DEFAULT_CONTEXT_LAYERS,
        "context_feedforward": DEFAULT_CONTEXT_FEEDFORWARD,
        "learned_positional_embedding": False,
        "permutation_invariant_context": True,
        "daughter_order_invariant": True,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
    }
