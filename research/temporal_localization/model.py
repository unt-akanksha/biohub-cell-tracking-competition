"""High-capacity temporal 3D node localizer for Biohub.

The model is deliberately independent of the public competition notebooks.  It
combines a ConvNeXt-style 3D encoder, an axial 2D morphology encoder, and frozen
graph-motion features.  It predicts only a bounded coordinate correction and
uncertainty; it cannot add/remove nodes or change lineage edges.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


FAMILY = "temporal_convnext_axial_node_localizer_v1"
INPUT_CHANNELS = 3
PATCH_SHAPE = (17, 17, 17)
HALF_EXTENT_UM = (12.0, 12.0, 12.0)
GRAPH_FEATURE_WIDTH = 12
CONVNEXT_DIMS = (128, 256, 512, 768)
CONVNEXT_DEPTHS = (3, 3, 12, 4)
AXIAL_DIMS = (96, 192, 384, 512)
MAXIMUM_CORRECTION_UM = 10.0
EXPECTED_PARAMETER_COUNT = 71_249_805


class LayerNormChannelsFirst(nn.Module):
    """LayerNorm over channels without tying statistics to spatial size."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            inputs.movedim(1, -1),
            (inputs.shape[1],),
            self.weight,
            self.bias,
        ).movedim(-1, 1)


class ConvNeXtBlock3D(nn.Module):
    """Large-kernel depthwise 3D block with channel MLP and layer scale."""

    def __init__(self, channels: int, *, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, hidden)
        self.contract = nn.Linear(hidden, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(inputs).movedim(1, -1)
        hidden = self.contract(F.gelu(self.expand(self.norm(hidden))))
        hidden = hidden * self.layer_scale
        return inputs + hidden.movedim(-1, 1)


class ConvNeXt3DEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = INPUT_CHANNELS,
        dims: Sequence[int] = CONVNEXT_DIMS,
        depths: Sequence[int] = CONVNEXT_DEPTHS,
    ) -> None:
        super().__init__()
        dims = tuple(int(value) for value in dims)
        depths = tuple(int(value) for value in depths)
        if len(dims) != 4 or len(depths) != 4 or min(*dims, *depths) <= 0:
            raise ValueError("ConvNeXt dims and depths must contain four positive values")
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, dims[0], kernel_size=3, padding=1, bias=False),
            LayerNormChannelsFirst(dims[0]),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (channels, depth) in enumerate(zip(dims, depths, strict=True)):
            self.stages.append(
                nn.Sequential(*(ConvNeXtBlock3D(channels) for _ in range(depth)))
            )
            if index < len(dims) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        LayerNormChannelsFirst(channels),
                        nn.Conv3d(channels, dims[index + 1], kernel_size=2, stride=2),
                    )
                )
        self.output_channels = dims[-1]

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(patches)
        for index, stage in enumerate(self.stages):
            hidden = stage(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        return hidden.mean(dim=(2, 3, 4))


class ResidualBlock2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = next(value for value in (16, 8, 4, 2, 1) if output_channels % value == 0)
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.main(inputs) + self.skip(inputs))


class AxialMorphologyEncoder(nn.Module):
    """Retain center/mean/max/std and learned Z-attention projections."""

    def __init__(self, input_channels: int = INPUT_CHANNELS) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.attention = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, 7, padding=3, groups=input_channels),
            nn.GELU(),
            nn.Conv3d(input_channels, input_channels, 1, groups=input_channels),
        )
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)
        projected_channels = input_channels * 5
        self.stem = nn.Sequential(
            nn.Conv2d(projected_channels, AXIAL_DIMS[0], 3, padding=1, bias=False),
            nn.GroupNorm(16, AXIAL_DIMS[0]),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        for index, channels in enumerate(AXIAL_DIMS):
            if index == 0:
                blocks.append(ResidualBlock2D(channels, channels))
            else:
                blocks.extend(
                    (
                        ResidualBlock2D(AXIAL_DIMS[index - 1], channels, stride=2),
                        ResidualBlock2D(channels, channels),
                    )
                )
        self.encoder = nn.Sequential(*blocks)
        self.output_channels = AXIAL_DIMS[-1]

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(patches), dim=2)
        axial = torch.cat(
            (
                patches.mean(dim=2),
                patches.amax(dim=2),
                (patches.var(dim=2, unbiased=False) + 1e-6).sqrt(),
                patches[:, :, patches.shape[2] // 2],
                (weights * patches).sum(dim=2),
            ),
            dim=1,
        )
        return self.encoder(self.stem(axial)).mean(dim=(2, 3))


class TemporalNodeLocalizationModel(nn.Module):
    """Predict offset, aleatoric scale, and whether a proposal is already safe."""

    def __init__(self) -> None:
        super().__init__()
        self.physical = ConvNeXt3DEncoder()
        self.axial = AxialMorphologyEncoder()
        self.graph = nn.Sequential(
            nn.LayerNorm(GRAPH_FEATURE_WIDTH),
            nn.Linear(GRAPH_FEATURE_WIDTH, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        feature_width = self.physical.output_channels + self.axial.output_channels + 128
        self.head = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, 1_024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1_024, 512),
            nn.GELU(),
            nn.Linear(512, 7),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, patches: torch.Tensor, graph_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if patches.ndim != 5 or patches.shape[1] != INPUT_CHANNELS:
            raise ValueError("patches must have shape (N, 3, Z, Y, X)")
        if graph_features.shape != (len(patches), GRAPH_FEATURE_WIDTH):
            raise ValueError("graph features must have shape (N, 12)")
        if patches.device != graph_features.device:
            raise ValueError("patches and graph features must share one device")
        features = torch.cat(
            (self.physical(patches), self.axial(patches), self.graph(graph_features)),
            dim=1,
        )
        raw = self.head(features)
        offsets = torch.tanh(raw[:, :3]) * MAXIMUM_CORRECTION_UM
        log_variance = raw[:, 3:6].clamp(-4.0, 4.0)
        safe_probability = torch.sigmoid(raw[:, 6])
        return offsets, log_variance, safe_probability


def localization_loss(
    predicted_offsets_um: torch.Tensor,
    log_variance: torch.Tensor,
    safe_probability: torch.Tensor,
    target_offsets_um: torch.Tensor,
    *,
    safe_radius_um: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if predicted_offsets_um.shape != target_offsets_um.shape or predicted_offsets_um.shape[-1] != 3:
        raise ValueError("localization offsets must have shape (N, 3)")
    if log_variance.shape != target_offsets_um.shape or safe_probability.shape != target_offsets_um.shape[:1]:
        raise ValueError("uncertainty outputs do not match the localization inventory")
    residual = predicted_offsets_um - target_offsets_um
    robust = F.smooth_l1_loss(predicted_offsets_um, target_offsets_um, beta=0.5)
    heteroscedastic = 0.5 * (torch.exp(-log_variance) * residual.square() + log_variance).mean()
    safe_target = (torch.linalg.vector_norm(target_offsets_um, dim=1) <= safe_radius_um).float()
    # Probability-form BCE is deliberately blocked by CUDA autocast. Keep the
    # public model contract probability-based for inference and compute this
    # small calibration term explicitly in FP32 outside the surrounding AMP
    # region used by training.
    with torch.autocast(device_type=safe_probability.device.type, enabled=False):
        calibration = F.binary_cross_entropy(
            safe_probability.float(), safe_target.float()
        )
    total = robust + 0.1 * heteroscedastic + 0.05 * calibration
    return total, {
        "robust": robust.detach(),
        "heteroscedastic": heteroscedastic.detach(),
        "calibration": calibration.detach(),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def architecture_contract() -> dict[str, object]:
    model = TemporalNodeLocalizationModel()
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"parameter inventory changed: {count} != {EXPECTED_PARAMETER_COUNT}")
    return {
        "family": FAMILY,
        "parameter_count": count,
        "input_channels": INPUT_CHANNELS,
        "patch_shape": list(PATCH_SHAPE),
        "half_extent_um": list(HALF_EXTENT_UM),
        "graph_feature_width": GRAPH_FEATURE_WIDTH,
        "maximum_correction_um": MAXIMUM_CORRECTION_UM,
        "topology_preserving": True,
        "node_count_preserving": True,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
    }
