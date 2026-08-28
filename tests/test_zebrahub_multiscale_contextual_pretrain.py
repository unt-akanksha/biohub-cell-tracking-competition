from __future__ import annotations

from pathlib import Path

from research.temporal_contrastive import (
    train_zebrahub_contextual_pretrain as generic,
)
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (
    EXPECTED_PARAMETER_COUNT,
    MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
    MultiscaleContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_zebrahub_multiscale_contextual_pretrain import (
    RUN_ID,
    configure,
    multiscale_family_metadata,
)


def test_multiscale_pretrainer_configures_a_distinct_two_gpu_model_lane() -> None:
    original = (
        generic.RUN_ID,
        generic.APPEARANCE_FAMILY,
        generic.MODEL_CLASS,
        generic.MODEL_PARAMETER_COUNT,
        generic.WORKER_SCRIPT_PATH,
        generic.family_metadata,
        generic.initialize_model,
    )
    try:
        configure()
        assert generic.RUN_ID == RUN_ID
        assert generic.APPEARANCE_FAMILY == MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY
        assert generic.MODEL_CLASS is MultiscaleContextualPairFusionAssociationModel
        assert generic.MODEL_PARAMETER_COUNT == EXPECTED_PARAMETER_COUNT
        assert generic.WORKER_SCRIPT_PATH.name == (
            "train_zebrahub_multiscale_contextual_pretrain.py"
        )
        metadata = generic.family_metadata()
        assert metadata == multiscale_family_metadata()
        assert metadata["architecture_parent"] == "temporal_contextual_pair_fusion_v3"
        assert metadata["public_code_copied"] is False
        assert metadata["public_predictions_copied"] is False
        assert metadata["public_leaderboard_used_for_selection"] is False
        assert generic.initialize_model.__name__ == "initialize_from_contextual_v3"
    finally:
        (
            generic.RUN_ID,
            generic.APPEARANCE_FAMILY,
            generic.MODEL_CLASS,
            generic.MODEL_PARAMETER_COUNT,
            generic.WORKER_SCRIPT_PATH,
            generic.family_metadata,
            generic.initialize_model,
        ) = original


def test_base_pretrainer_defaults_remain_frozen_v3() -> None:
    assert Path(generic.__file__).name == "train_zebrahub_contextual_pretrain.py"
    assert generic.RUN_ID == "zebrahub-contextual-pretrain-v1"
    assert generic.APPEARANCE_FAMILY == "temporal_contextual_pair_fusion_v3"
