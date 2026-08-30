"""Heavy daughter-order-invariant model for additive division recovery."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

try:
    from multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT as BACKBONE_PARAMETER_COUNT,
        MultiscaleContextualPairFusionAssociationModel,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT as BACKBONE_PARAMETER_COUNT,
        MultiscaleContextualPairFusionAssociationModel,
    )


RELATIONAL_DIVISION_FAMILY = "temporal_multiscale_relational_division_v1"
CENTER_COUNT = 3
TEMPORAL_CHANNELS = 3
RAW_GEOMETRY_WIDTH = 9
NORMALIZED_GEOMETRY_WIDTH = 18
DEFAULT_EMBEDDING_CHANNELS = 256
RELATIONAL_VISUAL_BLOCKS = 5
RELATIONAL_VISUAL_WIDTH = DEFAULT_EMBEDDING_CHANNELS * RELATIONAL_VISUAL_BLOCKS
BACKBONE_LOGIT_WIDTH = 3
RELATIONAL_FEATURE_WIDTH = (
    RELATIONAL_VISUAL_WIDTH + BACKBONE_LOGIT_WIDTH + NORMALIZED_GEOMETRY_WIDTH
)
RELATIONAL_HEAD_WIDTHS = (1024, 512, 128)


def _symmetric_geometry(raw: torch.Tensor) -> torch.Tensor:
    """Convert extractor geometry into daughter-order-invariant coordinates."""

    if raw.ndim != 2 or raw.shape[1] != RAW_GEOMETRY_WIDTH:
        raise ValueError(
            f"geometry must have shape (N, {RAW_GEOMETRY_WIDTH})"
        )
    proposed_distance = raw[:, 0]
    existing_distance = raw[:, 2]
    return torch.stack(
        (
            torch.minimum(proposed_distance, existing_distance),
            torch.maximum(proposed_distance, existing_distance),
            raw[:, 1],
            raw[:, 3],
            raw[:, 4],
            raw[:, 5],
            raw[:, 6],
            raw[:, 7],
            raw[:, 8],
        ),
        dim=1,
    )


def normalize_geometry(raw: torch.Tensor) -> torch.Tensor:
    symmetric = _symmetric_geometry(raw.float())
    missing = ~torch.isfinite(symmetric)
    values = torch.nan_to_num(symmetric, nan=0.0, posinf=0.0, neginf=0.0)
    scales = values.new_tensor((12.0, 12.0, 15.0, 8.0, 1.0, 1.0, 8.0, 8.0, 8.0))
    normalized = (values / scales).clamp(-4.0, 4.0)
    return torch.cat((normalized, missing.to(normalized.dtype)), dim=1)


class RelationalDivisionModel(nn.Module):
    """Shared 46.4M encoder plus symmetric parent/two-daughter reasoning."""

    def __init__(
        self,
        *,
        base_channels: int = 64,
        embedding_channels: int = DEFAULT_EMBEDDING_CHANNELS,
        projection_base_channels: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("relational dropout must be in [0, 1)")
        self.embedding_channels = int(embedding_channels)
        self.backbone = MultiscaleContextualPairFusionAssociationModel(
            base_channels=base_channels,
            embedding_channels=embedding_channels,
            projection_base_channels=projection_base_channels,
        )
        visual_width = RELATIONAL_VISUAL_BLOCKS * self.embedding_channels
        feature_width = visual_width + BACKBONE_LOGIT_WIDTH + NORMALIZED_GEOMETRY_WIDTH
        self.relational_feature_width = feature_width
        layers: list[nn.Module] = [nn.LayerNorm(feature_width)]
        input_width = feature_width
        for output_width in RELATIONAL_HEAD_WIDTHS:
            layers.extend(
                (
                    nn.Linear(input_width, output_width),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            input_width = output_width
        layers.append(nn.Linear(input_width, 1))
        self.relational_head = nn.Sequential(*layers)

    def forward(
        self, patches: torch.Tensor, geometry: torch.Tensor
    ) -> torch.Tensor:
        if patches.ndim != 6 or tuple(patches.shape[1:3]) != (
            CENTER_COUNT,
            TEMPORAL_CHANNELS,
        ):
            raise ValueError(
                "relational patches must have shape (N, 3 centers, 3 temporal, Z, Y, X)"
            )
        if geometry.shape != (len(patches), RAW_GEOMETRY_WIDTH):
            raise ValueError(
                f"geometry must have shape ({len(patches)}, {RAW_GEOMETRY_WIDTH})"
            )
        flattened = patches.reshape(-1, *patches.shape[2:])
        embeddings, division_logits = self.backbone(flattened)
        embeddings = embeddings.reshape(
            len(patches), CENTER_COUNT, self.embedding_channels
        )
        division_logits = division_logits.reshape(len(patches), CENTER_COUNT)
        parent = embeddings[:, 0]
        first_daughter = embeddings[:, 1]
        second_daughter = embeddings[:, 2]
        daughter_mean = 0.5 * (first_daughter + second_daughter)
        daughter_difference = torch.abs(first_daughter - second_daughter)
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
        daughter_logit_mean = division_logits[:, 1:].mean(dim=1)
        daughter_logit_difference = torch.abs(
            division_logits[:, 1] - division_logits[:, 2]
        )
        logit_features = torch.stack(
            (
                division_logits[:, 0],
                daughter_logit_mean,
                daughter_logit_difference,
            ),
            dim=1,
        )
        features = torch.cat(
            (visual, logit_features, normalize_geometry(geometry)), dim=1
        )
        if features.shape[1] != self.relational_feature_width:
            raise RuntimeError("relational feature width changed")
        return self.relational_head(features).squeeze(1)


def load_backbone_checkpoint(
    model: RelationalDivisionModel, state_dict: Mapping[str, torch.Tensor]
) -> None:
    if not isinstance(model, RelationalDivisionModel):
        raise TypeError("relational warm start requires a relational model")
    expected = model.backbone.state_dict()
    if set(state_dict) != set(expected):
        missing = sorted(set(expected) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected))
        raise ValueError(
            f"relational backbone keys changed: missing={missing}, unexpected={unexpected}"
        )
    mismatched = sorted(
        name
        for name, value in state_dict.items()
        if tuple(value.shape) != tuple(expected[name].shape)
    )
    if mismatched:
        raise ValueError(f"relational backbone shapes changed: {mismatched}")
    model.backbone.load_state_dict(dict(state_dict), strict=True)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def architecture_contract() -> dict[str, object]:
    model = RelationalDivisionModel()
    count = parameter_count(model)
    backbone_count = parameter_count(model.backbone)
    if backbone_count != BACKBONE_PARAMETER_COUNT:
        raise RuntimeError("relational backbone parameter inventory changed")
    return {
        "family": RELATIONAL_DIVISION_FAMILY,
        "parameter_count": count,
        "backbone_parameter_count": backbone_count,
        "relational_parameter_count": count - backbone_count,
        "center_count": CENTER_COUNT,
        "temporal_channels_per_center": TEMPORAL_CHANNELS,
        "geometry_width": NORMALIZED_GEOMETRY_WIDTH,
        "daughter_order_invariant": True,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
    }
