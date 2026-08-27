from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.temporal_contrastive import (
    train_dual_fold_contextual_pair_fusion as contextual,
)
from research.temporal_contrastive import train_dual_fold_pair_fusion as generic
from research.temporal_contrastive.appearance_family import (
    verify_appearance_metadata,
)
from research.temporal_contrastive.contextual_pair_fusion import (
    CONTEXTUAL_PAIR_FUSION_FAMILY,
    EXPECTED_PARAMETER_COUNT,
    ContextualPairFusionAssociationModel,
)


def test_contextual_trainer_metadata_satisfies_exact_family_contract() -> None:
    payload = {
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "run_id": contextual.RUN_ID,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "input_channels": 3,
        "temporal_frame_offsets": [-1, 0, 1],
        "checkpoint_weight_source": "optimizer-step exponential moving average",
        "ema_decay": 0.997,
        "division_prior_correction": "class-conditional importance weighting",
        "link_loss_policy": (
            "all-positive supervised contrastive mean-log-probability"
        ),
        "real_split_policy": (
            "global deterministic disjoint partition per embryo prefix"
        ),
        **contextual.contextual_family_metadata(),
    }

    assert verify_appearance_metadata(
        payload, require_training_run=True
    ) == CONTEXTUAL_PAIR_FUSION_FAMILY


def test_contextual_trainer_configures_reciprocal_engine_and_restores() -> None:
    names = (
        "RUN_ID",
        "APPEARANCE_FAMILY",
        "MODEL_CLASS",
        "EXPECTED_PARAMETER_COUNT",
        "WORKER_SCRIPT_PATH",
        "family_metadata",
        "pair_logits_for_transition",
        "pair_loss_for_transition",
        "validate_pair_model",
    )
    original = {name: getattr(generic, name) for name in names}
    try:
        contextual.configure()
        assert generic.RUN_ID == contextual.RUN_ID
        assert generic.APPEARANCE_FAMILY == CONTEXTUAL_PAIR_FUSION_FAMILY
        assert generic.MODEL_CLASS is ContextualPairFusionAssociationModel
        assert generic.EXPECTED_PARAMETER_COUNT == EXPECTED_PARAMETER_COUNT
        assert generic.WORKER_SCRIPT_PATH == Path(contextual.__file__).resolve()
        assert generic.family_metadata is contextual.contextual_family_metadata
        assert generic.pair_logits_for_transition is contextual.contextual_pair_adapter
        assert (
            generic.pair_loss_for_transition
            is contextual.contextual_bidirectional_pair_nll
        )
        assert generic.validate_pair_model is contextual.validate_contextual_model
    finally:
        for name, value in original.items():
            setattr(generic, name, value)


def test_contextual_pair_adapter_fails_closed_without_transition_volumes() -> None:
    with pytest.raises(ValueError, match="requires transition volumes"):
        contextual.contextual_pair_adapter(
            ContextualPairFusionAssociationModel(
                base_channels=8, embedding_channels=16
            ),
            torch.zeros((1, 16)),
            torch.zeros((1, 16)),
            torch.zeros((1,)),
            object(),
            (1.0, 1.0, 1.0),
            torch.device("cpu"),
            candidate_radius_um=32.0,
            pair_chunk_size=4_096,
            source_volume=np.zeros((3, 3, 3, 3), dtype=np.float32),
            target_volume=None,
        )


def test_contextual_trainer_requires_external_pretraining_evidence() -> None:
    contextual.require_external_pretraining(
        ["--orchestrate", "--initial-model-root", "/evidence"]
    )
    contextual.require_external_pretraining(
        ["--worker", "--initial-model-root=/evidence"]
    )
    with pytest.raises(ValueError, match="requires hash-bound ZebraHub"):
        contextual.require_external_pretraining(["--orchestrate"])


def test_transfer_gate_requires_broad_real_gain_without_forgetting() -> None:
    initial_real = {
        "composite": 0.70,
        "top1": 0.68,
        "mrr": 0.75,
        "division_top2": 0.40,
        "rows": 500,
        "division_rows": 50,
        "transitions": 48,
    }
    initial_synthetic = {
        "composite": 0.90,
        "top1": 0.91,
        "mrr": 0.94,
        "division_top2": 0.80,
        "rows": 1_500,
        "division_rows": 450,
        "transitions": 32,
    }
    candidate_real = {
        **initial_real,
        "composite": 0.72,
        "top1": 0.70,
        "mrr": 0.77,
        "division_top2": 0.42,
    }
    candidate_synthetic = {
        **initial_synthetic,
        "composite": 0.895,
        "top1": 0.905,
        "mrr": 0.935,
        "division_top2": 0.795,
    }

    gate = generic.finetuning_improvement_gate(
        initial_real,
        candidate_real,
        initial_synthetic,
        candidate_synthetic,
    )

    assert gate["passed"] is True
    assert gate["real_gains"]["composite"] == pytest.approx(0.02)
    assert gate["synthetic_gains"]["composite"] == pytest.approx(-0.005)
    assert generic.finetuning_improvement_gate(
        initial_real,
        {**candidate_real, "division_top2": 0.39},
        initial_synthetic,
        candidate_synthetic,
    )["passed"] is False
    assert generic.finetuning_improvement_gate(
        initial_real,
        candidate_real,
        initial_synthetic,
        {**candidate_synthetic, "composite": 0.88},
    )["passed"] is False
    assert generic.finetuning_improvement_gate(
        initial_real,
        {**candidate_real, "rows": 499},
        initial_synthetic,
        candidate_synthetic,
    )["passed"] is False
