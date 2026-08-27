"""Fail-closed contracts for temporal appearance model families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from appearance_blend import appearance_scores_for_movie
    from pair_fusion import (
        DEFAULT_PAIR_CHUNK_SIZE,
        EXPECTED_PARAMETER_COUNT as PAIR_FUSION_PARAMETER_COUNT,
        PAIR_FEATURE_WIDTH,
        PAIR_FUSION_FAMILY,
        PAIR_FUSION_POLICY,
        PAIR_HIDDEN_WIDTHS,
        PAIR_LOSS_POLICY,
        PAIR_PROJECTION_WIDTH,
        PhysicalPairFusionAssociationModel,
        pair_fusion_scores_for_movie,
    )
    from patch_model import PhysicalPatchAssociationModel
except ModuleNotFoundError:
    from research.temporal_contrastive.appearance_blend import (
        appearance_scores_for_movie,
    )
    from research.temporal_contrastive.pair_fusion import (
        DEFAULT_PAIR_CHUNK_SIZE,
        EXPECTED_PARAMETER_COUNT as PAIR_FUSION_PARAMETER_COUNT,
        PAIR_FEATURE_WIDTH,
        PAIR_FUSION_FAMILY,
        PAIR_FUSION_POLICY,
        PAIR_HIDDEN_WIDTHS,
        PAIR_LOSS_POLICY,
        PAIR_PROJECTION_WIDTH,
        PhysicalPairFusionAssociationModel,
        pair_fusion_scores_for_movie,
    )
    from research.temporal_contrastive.patch_model import (
        PhysicalPatchAssociationModel,
    )


COSINE_FAMILY = "temporal_cosine_v1"
COSINE_PARAMETER_COUNT = 19_221_954
PAIR_FUSION_EMBEDDING_LOSS_WEIGHT = 0.25
LINK_LOSS_POLICY = "all-positive supervised contrastive mean-log-probability"
FAMILIES = (COSINE_FAMILY, PAIR_FUSION_FAMILY)
TRAINING_RUN_BY_FAMILY = {
    COSINE_FAMILY: "temporal-patch-dual-fold-v1",
    PAIR_FUSION_FAMILY: "temporal-patch-pair-fusion-v2",
}
CALIBRATION_RUN_BY_FAMILY = {
    COSINE_FAMILY: "temporal-patch-dual-fold-blend-v1",
    PAIR_FUSION_FAMILY: "temporal-patch-pair-fusion-blend-v2",
}
PROCESSED_RUN_BY_FAMILY = {
    COSINE_FAMILY: "temporal-patch-dual-fold-processed-acceptance-v1",
    PAIR_FUSION_FAMILY: "temporal-patch-pair-fusion-processed-acceptance-v2",
}
CANDIDATE_FAMILY_BY_APPEARANCE = {
    COSINE_FAMILY: "trackastra_appearance_blend",
    PAIR_FUSION_FAMILY: "trackastra_pair_fusion_blend",
}
APPEARANCE_FAMILY_BY_CANDIDATE = {
    candidate: family for family, candidate in CANDIDATE_FAMILY_BY_APPEARANCE.items()
}
PARAMETER_COUNT_BY_FAMILY = {
    COSINE_FAMILY: COSINE_PARAMETER_COUNT,
    PAIR_FUSION_FAMILY: PAIR_FUSION_PARAMETER_COUNT,
}
COMMON_METADATA_KEYS = (
    "appearance_family",
    "parameter_count",
    "input_channels",
    "temporal_frame_offsets",
    "checkpoint_weight_source",
    "ema_decay",
    "division_prior_correction",
    "link_loss_policy",
    "real_split_policy",
)
PAIR_FUSION_METADATA_KEYS = (
    "pair_feature_width",
    "pair_projection_width",
    "pair_hidden_widths",
    "pair_fusion_policy",
    "pair_loss_policy",
    "embedding_auxiliary_loss_weight",
    "pair_chunk_size",
)


def appearance_family(payload: Mapping[str, Any]) -> str:
    """Resolve legacy v1 evidence explicitly and reject ambiguous schemas."""

    declared = payload.get("appearance_family")
    if declared is None:
        if int(payload.get("parameter_count", 0)) != COSINE_PARAMETER_COUNT:
            raise ValueError("appearance evidence omits its model family")
        return COSINE_FAMILY
    family = str(declared)
    if family not in FAMILIES:
        raise ValueError(f"unknown appearance model family: {family}")
    return family


def verify_appearance_metadata(
    payload: Mapping[str, Any],
    *,
    require_training_run: bool = False,
) -> str:
    """Verify exact architecture/loss metadata and return the family."""

    family = appearance_family(payload)
    if int(payload.get("parameter_count", 0)) != PARAMETER_COUNT_BY_FAMILY[family]:
        raise ValueError("appearance parameter count changed")
    if require_training_run and payload.get("run_id") != TRAINING_RUN_BY_FAMILY[family]:
        raise ValueError("appearance training run does not match its family")
    common_valid = bool(
        payload.get("input_channels") == 3
        and payload.get("temporal_frame_offsets") == [-1, 0, 1]
        and payload.get("checkpoint_weight_source")
        == "optimizer-step exponential moving average"
        and payload.get("ema_decay") == 0.997
        and payload.get("division_prior_correction")
        == "class-conditional importance weighting"
        and payload.get("link_loss_policy") == LINK_LOSS_POLICY
        and payload.get("real_split_policy")
        == "global deterministic disjoint partition per embryo prefix"
    )
    if not common_valid:
        raise ValueError("appearance common architecture contract changed")
    if family == PAIR_FUSION_FAMILY:
        pair_valid = bool(
            payload.get("appearance_family") == PAIR_FUSION_FAMILY
            and payload.get("pair_feature_width") == PAIR_FEATURE_WIDTH
            and payload.get("pair_projection_width") == PAIR_PROJECTION_WIDTH
            and payload.get("pair_hidden_widths") == list(PAIR_HIDDEN_WIDTHS)
            and payload.get("pair_fusion_policy") == PAIR_FUSION_POLICY
            and payload.get("pair_loss_policy") == PAIR_LOSS_POLICY
            and payload.get("embedding_auxiliary_loss_weight")
            == PAIR_FUSION_EMBEDDING_LOSS_WEIGHT
            and payload.get("pair_chunk_size") == DEFAULT_PAIR_CHUNK_SIZE
        )
        if not pair_valid:
            raise ValueError("pair-fusion architecture contract changed")
    return family


def appearance_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the exact verified family contract into downstream evidence."""

    family = verify_appearance_metadata(payload)
    keys = COMMON_METADATA_KEYS
    if family == PAIR_FUSION_FAMILY:
        keys += PAIR_FUSION_METADATA_KEYS
    result = {key: payload[key] for key in keys if key in payload}
    result["appearance_family"] = family
    return result


def candidate_appearance_family(candidate_family: str) -> str:
    try:
        return APPEARANCE_FAMILY_BY_CANDIDATE[str(candidate_family)]
    except KeyError as error:
        raise ValueError(
            f"unknown appearance candidate family: {candidate_family}"
        ) from error


def build_appearance_model(family: str) -> PhysicalPatchAssociationModel:
    if family == COSINE_FAMILY:
        return PhysicalPatchAssociationModel()
    if family == PAIR_FUSION_FAMILY:
        return PhysicalPairFusionAssociationModel()
    raise ValueError(f"unknown appearance model family: {family}")


def appearance_evidence_for_movie(
    family: str,
    model: PhysicalPatchAssociationModel,
    video: Any,
    embeddings: Any,
    division_logits: Any,
    pair_scores: Any,
) -> dict[int, Any]:
    if family == COSINE_FAMILY:
        if type(model) is not PhysicalPatchAssociationModel:
            raise TypeError("cosine appearance model type changed")
        return appearance_scores_for_movie(video, embeddings, pair_scores)
    if family == PAIR_FUSION_FAMILY:
        if not isinstance(model, PhysicalPairFusionAssociationModel):
            raise TypeError("pair-fusion appearance model type changed")
        return pair_fusion_scores_for_movie(
            model,
            video,
            embeddings,
            division_logits,
            pair_scores,
            candidate_radius_um=32.0,
            chunk_size=DEFAULT_PAIR_CHUNK_SIZE,
        )
    raise ValueError(f"unknown appearance model family: {family}")
