"""Project-authored multiscale appearance branch for a future contextual v4.

The frozen contextual v3 model remains unchanged.  This module adds a separate
high-capacity experiment: a native 3D residual path is fused with an efficient
2D path built from deterministic axial statistics and one learned axial
projection.  The design tests whether z-compressed morphology supplies
complementary association evidence without copying public notebook code,
weights, predictions, or leaderboard choices.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

try:
    from contextual_pair_fusion import ContextualPairFusionAssociationModel
except ModuleNotFoundError:
    from research.temporal_contrastive.contextual_pair_fusion import (
        ContextualPairFusionAssociationModel,
    )


MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY = (
    "temporal_multiscale_contextual_pair_fusion_v4"
)
MULTISCALE_PROJECTION_POLICY = (
    "per-temporal-channel axial mean-max-std-center-learned-attention fused "
    "with the native physical 3D encoder"
)
DEFAULT_PROJECTION_BASE_CHANNELS = 96
AXIAL_STATISTICS_PER_CHANNEL = 5
EXPECTED_PARAMETER_COUNT = 46_386_607


def _normalization_groups(channels: int) -> int:
    return next(value for value in (16, 8, 4, 2, 1) if channels % value == 0)


class ResidualBlock2D(nn.Module):
    """Small GroupNorm residual unit for the axial-projection branch."""

    def __init__(
        self, input_channels: int, output_channels: int, *, stride: int = 1
    ) -> None:
        super().__init__()
        if min(input_channels, output_channels, stride) <= 0:
            raise ValueError("2D residual block dimensions must be positive")
        groups = _normalization_groups(output_channels)
        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(inputs) + self.skip(inputs), inplace=True)


class AxialProjection2D(nn.Module):
    """Compress Z while retaining fixed and learned morphology summaries."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("axial projection input channels must be positive")
        self.input_channels = int(input_channels)
        groups = _normalization_groups(self.input_channels)
        self.attention_logits = nn.Sequential(
            nn.Conv3d(
                self.input_channels,
                self.input_channels,
                kernel_size=3,
                padding=1,
                groups=self.input_channels,
                bias=False,
            ),
            nn.GroupNorm(groups, self.input_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                self.input_channels,
                self.input_channels,
                kernel_size=1,
                groups=self.input_channels,
                bias=True,
            ),
        )
        # Begin as a uniform physical projection. Training can move attention
        # along Z independently for each temporal input channel.
        nn.init.zeros_(self.attention_logits[-1].weight)
        nn.init.zeros_(self.attention_logits[-1].bias)

    @property
    def output_channels(self) -> int:
        return self.input_channels * AXIAL_STATISTICS_PER_CHANNEL

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5 or patches.shape[1] != self.input_channels:
            raise ValueError(
                "axial patches must have shape "
                f"(N, {self.input_channels}, Z, Y, X)"
            )
        if patches.shape[2] <= 0:
            raise ValueError("axial patches must contain at least one Z plane")
        attention = torch.softmax(self.attention_logits(patches), dim=2)
        learned = (attention * patches).sum(dim=2)
        mean = patches.mean(dim=2)
        maximum = patches.amax(dim=2)
        standard_deviation = (patches.var(dim=2, unbiased=False) + 1e-6).sqrt()
        center = patches[:, :, patches.shape[2] // 2]
        result = torch.cat(
            (mean, maximum, standard_deviation, center, learned), dim=1
        )
        if result.shape[1] != self.output_channels:
            raise RuntimeError("axial projection channel contract changed")
        return result


class MultiscaleContextualPairFusionAssociationModel(
    ContextualPairFusionAssociationModel
):
    """Contextual edge model with fused 3D and z-compressed 2D morphology."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        base_channels: int = 64,
        embedding_channels: int = 256,
        projection_base_channels: int = DEFAULT_PROJECTION_BASE_CHANNELS,
    ) -> None:
        if projection_base_channels <= 0:
            raise ValueError("projection base channels must be positive")
        super().__init__(
            input_channels=input_channels,
            base_channels=base_channels,
            embedding_channels=embedding_channels,
        )
        projection_base = int(projection_base_channels)
        self.projection_base_channels = projection_base
        self.axial_projection = AxialProjection2D(input_channels)
        projection_groups = _normalization_groups(projection_base)
        self.projection_stem = nn.Sequential(
            nn.Conv2d(
                self.axial_projection.output_channels,
                projection_base,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(projection_groups, projection_base),
            nn.SiLU(inplace=True),
        )
        self.projection_encoder = nn.Sequential(
            ResidualBlock2D(projection_base, projection_base),
            ResidualBlock2D(projection_base, projection_base * 2, stride=2),
            ResidualBlock2D(projection_base * 2, projection_base * 2),
            ResidualBlock2D(projection_base * 2, projection_base * 4, stride=2),
            ResidualBlock2D(projection_base * 4, projection_base * 4),
            ResidualBlock2D(projection_base * 4, projection_base * 8, stride=2),
            ResidualBlock2D(projection_base * 8, projection_base * 8),
        )
        projected_channels = projection_base * 8
        self.projected_feature_channels = projected_channels
        appearance_hidden = max(embedding_channels * 2, projection_base * 4)
        self.appearance_adapter = nn.Sequential(
            nn.LayerNorm(projected_channels),
            nn.Linear(projected_channels, appearance_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(appearance_hidden, embedding_channels),
        )
        division_hidden = max(base_channels * 2, projection_base * 2)
        self.division_adapter = nn.Sequential(
            nn.LayerNorm(projected_channels),
            nn.Linear(projected_channels, division_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(division_hidden, 1),
        )

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.ndim != 5 or patches.shape[1] != self.input_channels:
            raise ValueError(
                f"patches must have shape (N, {self.input_channels}, Z, Y, X)"
            )
        physical_embeddings, physical_divisions = super().forward(patches)
        axial = self.axial_projection(patches)
        projected = self.projection_encoder(self.projection_stem(axial)).mean(
            dim=(2, 3)
        )
        if projected.shape[1] != self.projected_feature_channels:
            raise RuntimeError("multiscale projected feature width changed")
        embeddings = F.normalize(
            physical_embeddings + self.appearance_adapter(projected),
            p=2,
            dim=1,
            eps=1e-8,
        )
        divisions = physical_divisions + self.division_adapter(projected).squeeze(1)
        return embeddings, divisions


def load_contextual_v3_warm_start(
    model: MultiscaleContextualPairFusionAssociationModel,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Strict-load every v3 tensor and start both new residuals at zero.

    The shared 3D encoder, projection, division head, logit scale, edge-token
    block, and contextual edge head must all be present with exact shapes. Only
    the new axial branch and its adapters may be absent from a v3 checkpoint.
    """

    if not isinstance(model, MultiscaleContextualPairFusionAssociationModel):
        raise TypeError("v3 warm start requires the multiscale contextual model")
    current = model.state_dict()
    unexpected = sorted(set(state_dict) - set(current))
    mismatched = sorted(
        key
        for key, value in state_dict.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    )
    if unexpected or mismatched:
        raise ValueError(
            "contextual v3 warm-start keys or tensor shapes changed: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    result = model.load_state_dict(dict(state_dict), strict=False)
    allowed_prefixes = (
        "axial_projection.",
        "projection_stem.",
        "projection_encoder.",
        "appearance_adapter.",
        "division_adapter.",
    )
    invalid_missing = sorted(
        key
        for key in result.missing_keys
        if not key.startswith(allowed_prefixes)
    )
    if result.unexpected_keys or invalid_missing:
        raise ValueError(
            "contextual v3 warm start is incomplete: "
            f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
        )
    with torch.no_grad():
        model.appearance_adapter[-1].weight.zero_()
        model.appearance_adapter[-1].bias.zero_()
        model.division_adapter[-1].weight.zero_()
        model.division_adapter[-1].bias.zero_()
    return tuple(sorted(result.missing_keys))


def parameter_count(model: nn.Module) -> int:
    """Return the trainable-plus-buffer-free parameter inventory."""

    return sum(parameter.numel() for parameter in model.parameters())


def architecture_contract() -> dict[str, object]:
    """Describe the immutable default v4 ablation architecture."""

    model = MultiscaleContextualPairFusionAssociationModel()
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"multiscale parameter inventory changed: {count} != "
            f"{EXPECTED_PARAMETER_COUNT}"
        )
    return {
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "input_channels": 3,
        "base_channels": 64,
        "embedding_channels": 256,
        "projection_base_channels": DEFAULT_PROJECTION_BASE_CHANNELS,
        "axial_statistics_per_channel": AXIAL_STATISTICS_PER_CHANNEL,
        "projection_policy": MULTISCALE_PROJECTION_POLICY,
        "parameter_count": count,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
    }
