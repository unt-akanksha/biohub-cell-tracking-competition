#!/usr/bin/env python
"""Pretrain the separate high-capacity multiscale contextual v4 lane.

This wrapper reuses the frozen ZebraHub split, reciprocal loss, augmentation,
selection/audit gates, two-GPU isolation, and no-submission boundary from v3.
Only the model factory and additive architecture metadata change. It must use a
distinct run and output directory and cannot replace or reinterpret v3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

try:
    import train_zebrahub_contextual_pretrain as generic
    from multiscale_contextual_pair_fusion import (
        AXIAL_STATISTICS_PER_CHANNEL,
        DEFAULT_PROJECTION_BASE_CHANNELS,
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MULTISCALE_PROJECTION_POLICY,
        MultiscaleContextualPairFusionAssociationModel,
        load_contextual_v3_warm_start,
    )
    from contextual_pair_fusion import ContextualPairFusionAssociationModel
    import train_dual_fold_contextual_pair_fusion as contextual_transfer
except ModuleNotFoundError:
    from research.temporal_contrastive import (
        train_zebrahub_contextual_pretrain as generic,
    )
    from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
        AXIAL_STATISTICS_PER_CHANNEL,
        DEFAULT_PROJECTION_BASE_CHANNELS,
        EXPECTED_PARAMETER_COUNT,
        MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        MULTISCALE_PROJECTION_POLICY,
        MultiscaleContextualPairFusionAssociationModel,
        load_contextual_v3_warm_start,
    )
    from research.temporal_contrastive.contextual_pair_fusion import (
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive import (
        train_dual_fold_contextual_pair_fusion as contextual_transfer,
    )


RUN_ID = "zebrahub-multiscale-contextual-pretrain-v1"


def multiscale_family_metadata() -> dict[str, object]:
    return {
        "architecture_parent": "temporal_contextual_pair_fusion_v3",
        "projection_base_channels": DEFAULT_PROJECTION_BASE_CHANNELS,
        "axial_statistics_per_channel": AXIAL_STATISTICS_PER_CHANNEL,
        "multiscale_projection_policy": MULTISCALE_PROJECTION_POLICY,
        "shared_contextual_edge_head": True,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
    }


def initialize_from_contextual_v3(
    model: torch.nn.Module,
    args: object,
    fold: str,
) -> dict[str, object]:
    if not isinstance(model, MultiscaleContextualPairFusionAssociationModel):
        raise TypeError("multiscale pretraining model factory changed")
    initial_root = getattr(args, "initial_model_root", None)
    if initial_root is None:
        raise ValueError("multiscale v4 requires --initial-model-root from accepted v3")
    contextual_transfer.configure()
    control = ContextualPairFusionAssociationModel()
    evidence = contextual_transfer.generic.load_initial_model(
        control, Path(initial_root), fold
    )
    missing = load_contextual_v3_warm_start(model, control.state_dict())
    missing_hash = hashlib.sha256(
        json.dumps(list(missing), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **evidence,
        "policy": "accepted contextual v3 plus zero-residual multiscale expansion",
        "source_family": "temporal_contextual_pair_fusion_v3",
        "target_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "new_parameter_keys": len(missing),
        "new_parameter_keys_sha256": missing_hash,
        "initial_predictions_numerically_preserved": True,
    }


def configure() -> None:
    generic.RUN_ID = RUN_ID
    generic.APPEARANCE_FAMILY = MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
    generic.MODEL_CLASS = MultiscaleContextualPairFusionAssociationModel
    generic.MODEL_PARAMETER_COUNT = EXPECTED_PARAMETER_COUNT
    generic.WORKER_SCRIPT_PATH = Path(__file__).resolve()
    generic.family_metadata = multiscale_family_metadata
    generic.initialize_model = initialize_from_contextual_v3


def main() -> None:
    if not any(
        value == "--initial-model-root"
        or value.startswith("--initial-model-root=")
        for value in sys.argv[1:]
    ):
        raise ValueError(
            "multiscale v4 requires the accepted contextual v3 pretraining root"
        )
    configure()
    generic.main()


if __name__ == "__main__":
    main()
