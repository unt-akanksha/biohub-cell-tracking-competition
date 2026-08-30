"""Calibration-free inference helpers for graph-context division models."""

from __future__ import annotations

from typing import Any

import torch

try:
    from graph_context_division_model import GraphContextDivisionModel
    from relational_division_inference import calibration_free_parent_scores
except ModuleNotFoundError:
    from research.temporal_contrastive.graph_context_division_model import (
        GraphContextDivisionModel,
    )
    from research.temporal_contrastive.relational_division_inference import (
        calibration_free_parent_scores,
    )


EXPECTED_PARAMETER_COUNT = 74_732_308


@torch.inference_mode()
def member_scores(
    models: list[GraphContextDivisionModel],
    patches: torch.Tensor,
    geometry: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    if not models or batch_size <= 0:
        raise ValueError("graph-context inference requires models and a positive batch")
    if not (
        len(patches) == len(geometry) == len(context) == len(context_mask)
    ):
        raise ValueError("graph-context inference arrays are not aligned")
    outputs: list[torch.Tensor] = []
    for model in models:
        model.eval()
        pieces = []
        for start in range(0, len(patches), batch_size):
            stop = start + batch_size
            batch = [
                value[start:stop].to(device=device, non_blocking=True)
                for value in (patches, geometry, context, context_mask)
            ]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pieces.append(model(*batch).float().cpu())
        outputs.append(torch.cat(pieces))
    return outputs


def calibration_free_scores(
    member_values: list[torch.Tensor], parent_ids: list[int]
) -> dict[int, float]:
    if any(len(values) != len(parent_ids) for values in member_values):
        raise ValueError("graph-context member scores are not candidate-aligned")
    mappings: list[dict[int, float]] = []
    for values in member_values:
        mappings.append(
            {
                int(parent_id): float(value)
                for parent_id, value in zip(parent_ids, values.tolist(), strict=True)
            }
        )
    return calibration_free_parent_scores(mappings, parent_ids)


def public_contract() -> dict[str, Any]:
    return {
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "absolute_threshold_used": False,
        "member_weights_searched": False,
        "model_subset_searched": False,
        "public_leaderboard_used_for_selection": False,
    }
