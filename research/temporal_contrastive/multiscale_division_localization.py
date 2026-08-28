"""High-capacity, physical-grid division localization derived from v4.

The association checkpoint is copied into a separate model. A zero-initialized
localization adapter learns to undo artificial integer offsets of externally
annotated node-centered patches. Association inference remains byte-for-byte
separate; this model may only donate bounded coordinates for complete predicted
division events.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

try:
    from multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT as V4_PARAMETER_COUNT,
        MultiscaleContextualPairFusionAssociationModel,
    )
except ModuleNotFoundError:
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        EXPECTED_PARAMETER_COUNT as V4_PARAMETER_COUNT,
        MultiscaleContextualPairFusionAssociationModel,
    )


DIVISION_LOCALIZATION_FAMILY = "multiscale_division_localization_v1"
SHARD_PATCH_SIZE = 17
LOCALIZATION_PATCH_SIZE = 13
PHYSICAL_GRID_SPACING_UM = 1.0
MAXIMUM_JITTER_GRID_STEPS = 2
MAXIMUM_CORRECTION_UM = 2.0
LOCALIZATION_FEATURE_WIDTH = 1_280
LOCALIZATION_HIDDEN_WIDTHS = (1_024, 256)
EXPECTED_PARAMETER_COUNT = 47_964_082


def integer_jitter_crops(
    patches: torch.Tensor, shifts_zyx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 13-cubed crops whose centers are shifted from known centers.

    The 17-cubed source patches span -8..8 µm with a one-micron grid. A shift
    in ``[-2, 2]`` therefore leaves a full 13-cubed crop without padding. The
    returned target is the physical correction back to the annotated center.
    """

    if patches.ndim != 5 or tuple(patches.shape[-3:]) != (
        SHARD_PATCH_SIZE,
        SHARD_PATCH_SIZE,
        SHARD_PATCH_SIZE,
    ):
        raise ValueError("patches must have shape (N, C, 17, 17, 17)")
    if shifts_zyx.shape != (len(patches), 3) or shifts_zyx.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("shifts must be an (N, 3) integer tensor")
    if shifts_zyx.device != patches.device:
        raise ValueError("patches and shifts must share one device")
    if torch.any(torch.abs(shifts_zyx) > MAXIMUM_JITTER_GRID_STEPS):
        raise ValueError("jitter exceeds the padding-free crop support")
    base = (SHARD_PATCH_SIZE - LOCALIZATION_PATCH_SIZE) // 2
    axis = torch.arange(LOCALIZATION_PATCH_SIZE, device=patches.device)
    z_rows = base + shifts_zyx[:, 0, None] + axis[None]
    y_rows = base + shifts_zyx[:, 1, None] + axis[None]
    x_rows = base + shifts_zyx[:, 2, None] + axis[None]
    channels = patches.shape[1]
    cropped = torch.gather(
        patches,
        2,
        z_rows[:, None, :, None, None].expand(
            -1, channels, -1, SHARD_PATCH_SIZE, SHARD_PATCH_SIZE
        ),
    )
    cropped = torch.gather(
        cropped,
        3,
        y_rows[:, None, None, :, None].expand(
            -1, channels, LOCALIZATION_PATCH_SIZE, -1, SHARD_PATCH_SIZE
        ),
    )
    cropped = torch.gather(
        cropped,
        4,
        x_rows[:, None, None, None, :].expand(
            -1, channels, LOCALIZATION_PATCH_SIZE, LOCALIZATION_PATCH_SIZE, -1
        ),
    )
    targets_um = -shifts_zyx.to(dtype=patches.dtype) * PHYSICAL_GRID_SPACING_UM
    return cropped, targets_um


class MultiscaleDivisionLocalizationModel(
    MultiscaleContextualPairFusionAssociationModel
):
    """V4 3D+axial backbone with a separate bounded coordinate adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.localization_head = nn.Sequential(
            nn.LayerNorm(LOCALIZATION_FEATURE_WIDTH),
            nn.Linear(LOCALIZATION_FEATURE_WIDTH, LOCALIZATION_HIDDEN_WIDTHS[0]),
            nn.SiLU(inplace=True),
            nn.Linear(
                LOCALIZATION_HIDDEN_WIDTHS[0], LOCALIZATION_HIDDEN_WIDTHS[1]
            ),
            nn.SiLU(inplace=True),
            nn.Linear(LOCALIZATION_HIDDEN_WIDTHS[1], 3),
        )
        nn.init.zeros_(self.localization_head[-1].weight)
        nn.init.zeros_(self.localization_head[-1].bias)

    def localization_features(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5 or patches.shape[1] != self.input_channels:
            raise ValueError(
                f"patches must have shape (N, {self.input_channels}, Z, Y, X)"
            )
        physical = self.encoder(self.stem(patches)).mean(dim=(2, 3, 4))
        axial = self.axial_projection(patches)
        projected = self.projection_encoder(self.projection_stem(axial)).mean(
            dim=(2, 3)
        )
        features = torch.cat((physical, projected), dim=1)
        if features.shape[1] != LOCALIZATION_FEATURE_WIDTH:
            raise RuntimeError("localization feature inventory changed")
        return features

    def localization_offsets_um(self, patches: torch.Tensor) -> torch.Tensor:
        raw = self.localization_head(self.localization_features(patches))
        return torch.tanh(raw) * MAXIMUM_CORRECTION_UM


def load_multiscale_v4_warm_start(
    model: MultiscaleDivisionLocalizationModel,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Strict-load a v4 checkpoint while allowing only the new adapter."""

    if not isinstance(model, MultiscaleDivisionLocalizationModel):
        raise TypeError("localization warm start requires the v1 localization model")
    current = model.state_dict()
    unexpected = sorted(set(state_dict) - set(current))
    mismatched = sorted(
        key
        for key, value in state_dict.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    )
    if unexpected or mismatched:
        raise ValueError(
            "v4 localization warm-start tensors changed: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    loaded = model.load_state_dict(dict(state_dict), strict=False)
    invalid_missing = sorted(
        key
        for key in loaded.missing_keys
        if not key.startswith("localization_head.")
    )
    if loaded.unexpected_keys or invalid_missing:
        raise ValueError(
            "v4 localization warm start is incomplete: "
            f"missing={invalid_missing}, unexpected={loaded.unexpected_keys}"
        )
    return tuple(sorted(loaded.missing_keys))


def localization_loss(
    predicted_offsets_um: torch.Tensor, target_offsets_um: torch.Tensor
) -> torch.Tensor:
    if predicted_offsets_um.shape != target_offsets_um.shape or (
        predicted_offsets_um.ndim != 2 or predicted_offsets_um.shape[1] != 3
    ):
        raise ValueError("predicted and target offsets must have shape (N, 3)")
    if not torch.isfinite(predicted_offsets_um).all() or not torch.isfinite(
        target_offsets_um
    ).all():
        raise ValueError("localization offsets must be finite")
    return F.smooth_l1_loss(
        predicted_offsets_um, target_offsets_um, beta=0.5, reduction="mean"
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def architecture_contract() -> dict[str, object]:
    model = MultiscaleDivisionLocalizationModel()
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"localization parameter inventory changed: {count} != "
            f"{EXPECTED_PARAMETER_COUNT}"
        )
    return {
        "family": DIVISION_LOCALIZATION_FAMILY,
        "v4_parameter_count": V4_PARAMETER_COUNT,
        "parameter_count": count,
        "source_patch_size": SHARD_PATCH_SIZE,
        "localization_patch_size": LOCALIZATION_PATCH_SIZE,
        "physical_grid_spacing_um": PHYSICAL_GRID_SPACING_UM,
        "maximum_jitter_grid_steps": MAXIMUM_JITTER_GRID_STEPS,
        "maximum_correction_um_per_axis": MAXIMUM_CORRECTION_UM,
        "prediction_preserving_initialization": True,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
    }

