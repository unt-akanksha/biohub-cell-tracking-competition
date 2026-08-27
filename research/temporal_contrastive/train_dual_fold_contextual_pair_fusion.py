#!/usr/bin/env python
"""Train the project-authored contextual pair-fusion v3 on exactly two GPUs.

This wrapper configures the reciprocal v2 training engine with the v3 model,
image-transition adapter, and frozen evidence contract.  It creates model
evidence only and has no competition submission path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

try:
    import train_dual_fold_pair_fusion as generic
    from contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FEATURE_WIDTH,
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        CONTEXTUAL_PAIR_LOSS_POLICY,
        CONTEXTUAL_PAIR_POLICY,
        EDGE_HEAD_HIDDEN_WIDTHS,
        EDGE_SET_FEATURE_WIDTH,
        EDGE_TOKEN_WIDTH,
        EXPECTED_PARAMETER_COUNT,
        TRANSITION_CONTEXT_POLICY,
        ContextualPairFusionAssociationModel,
    )
    from contextual_training import (
        contextual_pair_logits_for_transition,
        validate_contextual_model,
    )
    from transition_context import CANDIDATE_CONTEXT_WIDTH
except ModuleNotFoundError:
    from research.temporal_contrastive import (
        train_dual_fold_pair_fusion as generic,
    )
    from research.temporal_contrastive.contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FEATURE_WIDTH,
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        CONTEXTUAL_PAIR_LOSS_POLICY,
        CONTEXTUAL_PAIR_POLICY,
        EDGE_HEAD_HIDDEN_WIDTHS,
        EDGE_SET_FEATURE_WIDTH,
        EDGE_TOKEN_WIDTH,
        EXPECTED_PARAMETER_COUNT,
        TRANSITION_CONTEXT_POLICY,
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.contextual_training import (
        contextual_pair_logits_for_transition,
        validate_contextual_model,
    )
    from research.temporal_contrastive.transition_context import (
        CANDIDATE_CONTEXT_WIDTH,
    )


RUN_ID = "temporal-contextual-pair-fusion-v3"


def contextual_family_metadata() -> dict[str, object]:
    """Return the exact fail-closed v3 architecture and loss contract."""

    return {
        "candidate_context_width": CANDIDATE_CONTEXT_WIDTH,
        "contextual_pair_feature_width": CONTEXTUAL_PAIR_FEATURE_WIDTH,
        "edge_token_width": EDGE_TOKEN_WIDTH,
        "edge_set_feature_width": EDGE_SET_FEATURE_WIDTH,
        "edge_head_hidden_widths": list(EDGE_HEAD_HIDDEN_WIDTHS),
        "contextual_pair_policy": CONTEXTUAL_PAIR_POLICY,
        "transition_context_policy": TRANSITION_CONTEXT_POLICY,
        "pair_loss_policy": CONTEXTUAL_PAIR_LOSS_POLICY,
        "embedding_auxiliary_loss_weight": (
            generic.EMBEDDING_AUXILIARY_LOSS_WEIGHT
        ),
        "pair_chunk_size": generic.DEFAULT_PAIR_CHUNK_SIZE,
    }


def contextual_pair_adapter(
    model: ContextualPairFusionAssociationModel,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    source_division_logits: torch.Tensor,
    batch: generic.base.TransitionBatch,
    voxel_size_zyx_um: tuple[float, float, float],
    device: torch.device,
    *,
    candidate_radius_um: float,
    pair_chunk_size: int,
    source_volume: np.ndarray | None = None,
    target_volume: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if source_volume is None or target_volume is None:
        raise ValueError("contextual pair training requires transition volumes")
    logits, candidates, positives, _context = (
        contextual_pair_logits_for_transition(
            model,
            source_embeddings,
            target_embeddings,
            source_division_logits,
            batch,
            source_volume,
            target_volume,
            voxel_size_zyx_um,
            device,
            candidate_radius_um=candidate_radius_um,
            pair_chunk_size=pair_chunk_size,
        )
    )
    return logits, candidates, positives


def configure() -> None:
    """Install v3 components before orchestrator or isolated-worker startup."""

    generic.RUN_ID = RUN_ID
    generic.APPEARANCE_FAMILY = CONTEXTUAL_PAIR_FUSION_FAMILY
    generic.MODEL_CLASS = ContextualPairFusionAssociationModel
    generic.EXPECTED_PARAMETER_COUNT = EXPECTED_PARAMETER_COUNT
    generic.WORKER_SCRIPT_PATH = Path(__file__).resolve()
    generic.family_metadata = contextual_family_metadata
    generic.pair_logits_for_transition = contextual_pair_adapter
    generic.validate_pair_model = validate_contextual_model


def require_external_pretraining(argv: list[str]) -> None:
    if not any(
        value == "--initial-model-root"
        or value.startswith("--initial-model-root=")
        for value in argv
    ):
        raise ValueError(
            "contextual v3 requires hash-bound ZebraHub external pretraining"
        )


def main() -> None:
    require_external_pretraining(sys.argv[1:])
    configure()
    generic.main()


if __name__ == "__main__":
    main()
