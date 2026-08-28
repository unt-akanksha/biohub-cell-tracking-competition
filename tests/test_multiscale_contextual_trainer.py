from __future__ import annotations

from pathlib import Path

from research.temporal_contrastive import train_dual_fold_pair_fusion as generic
from research.temporal_contrastive.appearance_family import (
    build_appearance_model,
    candidate_appearance_family,
    verify_appearance_metadata,
)
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive import (
    train_dual_fold_multiscale_contextual_pair_fusion as multiscale,
)


def test_multiscale_transfer_configures_distinct_contextual_engine() -> None:
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
        "load_initial_model",
    )
    original = {name: getattr(generic, name) for name in names}
    try:
        multiscale.configure()
        assert generic.RUN_ID == multiscale.RUN_ID
        assert generic.APPEARANCE_FAMILY == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        assert generic.MODEL_CLASS is MultiscaleContextualPairFusionAssociationModel
        assert generic.EXPECTED_PARAMETER_COUNT == EXPECTED_PARAMETER_COUNT
        assert generic.WORKER_SCRIPT_PATH == Path(multiscale.__file__).resolve()
        assert generic.family_metadata is multiscale.multiscale_contextual_family_metadata
        assert generic.load_initial_model is multiscale.load_multiscale_initial_model
        metadata = generic.family_metadata()
        assert metadata["architecture_parent"] == "temporal_contextual_pair_fusion_v3"
        assert metadata["candidate_context_width"] == 18
        assert metadata["public_code_copied"] is False
        assert metadata["public_predictions_copied"] is False
        assert metadata["public_leaderboard_used_for_selection"] is False
    finally:
        for name, value in original.items():
            setattr(generic, name, value)


def test_multiscale_transfer_refuses_random_initialization() -> None:
    model = MultiscaleContextualPairFusionAssociationModel(
        base_channels=8,
        embedding_channels=16,
        projection_base_channels=8,
    )
    try:
        multiscale.load_multiscale_initial_model(model, None, "target_44b6")
    except ValueError as error:
        assert "requires hash-bound" in str(error)
    else:
        raise AssertionError("multiscale transfer accepted random initialization")


def test_multiscale_metadata_is_fail_closed_and_buildable() -> None:
    payload = {
        "appearance_family": MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "run_id": multiscale.RUN_ID,
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
        **multiscale.multiscale_contextual_family_metadata(),
    }

    assert verify_appearance_metadata(
        payload, require_training_run=True
    ) == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
    assert isinstance(
        build_appearance_model(MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY),
        MultiscaleContextualPairFusionAssociationModel,
    )
    assert candidate_appearance_family(
        "trackastra_multiscale_contextual_pair_fusion_blend"
    ) == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
    payload["projection_base_channels"] = 64
    try:
        verify_appearance_metadata(payload, require_training_run=True)
    except ValueError as error:
        assert "multiscale contextual" in str(error)
    else:
        raise AssertionError("mutated multiscale architecture was accepted")
