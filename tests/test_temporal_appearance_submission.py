from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.temporal_contrastive.dual_fold_appearance_submission import (
    APPEARANCE_NODE_COST,
    DEFAULT_INFERENCE_HARD_STOP_SECONDS,
    KAGGLE_GPU_NOTEBOOK_MAX_SECONDS,
    PAIR_FUSION_PAIR_COST,
    appearance_movie_inference_weight,
    load_acceptance,
    orchestrate,
    parser,
)
from research.trackastra_graph.dual_fold_processed_acceptance import (
    FROZEN_ASSOCIATION_CONFIGURATION,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo
from research.submission_sharding import MINIMUM_NOTEBOOK_RUNTIME_RESERVE_SECONDS
import numpy as np


FOLDS = ("target_44b6", "target_6bba")


def test_candidate_builder_preserves_two_hour_kaggle_runtime_reserve() -> None:
    default = parser().get_default("hard_stop_seconds")

    assert default == DEFAULT_INFERENCE_HARD_STOP_SECONDS == 36_000
    assert KAGGLE_GPU_NOTEBOOK_MAX_SECONDS - default == (
        MINIMUM_NOTEBOOK_RUNTIME_RESERVE_SECONDS
    )


def test_candidate_builder_rejects_runtime_reserve_override() -> None:
    with pytest.raises(ValueError, match="finalization reserve"):
        orchestrate(SimpleNamespace(hard_stop_seconds=36_001))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_fixture(tmp_path: Path):
    trackastra_dirs = {}
    appearance_models = {}
    models = {}
    appearances = {}
    blends = {}
    for index, fold in enumerate(FOLDS, start=1):
        trackastra_dir = tmp_path / fold / "trackastra"
        trackastra_dir.mkdir(parents=True)
        trackastra_model = trackastra_dir / "model.pt"
        trackastra_model.write_bytes(f"track-{fold}".encode())
        appearance_model = tmp_path / fold / "appearance.pt"
        appearance_model.write_bytes(f"appearance-{fold}".encode())
        trackastra_dirs[fold] = trackastra_dir
        appearance_models[fold] = appearance_model
        models[fold] = {"model_sha256": digest(trackastra_model), "best_step": 100}
        appearances[fold] = {
            "model_sha256": digest(appearance_model),
            "best_step": 200,
            "parameter_count": 19_221_954,
            "input_channels": 3,
            "temporal_frame_offsets": [-1, 0, 1],
            "checkpoint_weight_source": "optimizer-step exponential moving average",
            "ema_decay": 0.997,
            "division_prior_correction": "class-conditional importance weighting",
            "link_loss_policy": "all-positive supervised contrastive mean-log-probability",
            "real_split_policy": "global deterministic disjoint partition per embryo prefix",
        }
        blends[fold] = {
            "appearance_weight": index * 0.1,
            "division_weight": index * 0.05,
            "ensemble_mode": "target_only",
            "appearance_temperature": 0.10,
        }
    evidence = tmp_path / "acceptance.json"
    evidence.write_text(
        json.dumps(
            {
                "status": "accepted",
                "evaluation_kind": "exact_processed_dual_fold_acceptance",
                "exact_processed_gate_passed": True,
                "candidate_family": "trackastra_appearance_blend",
                "public_leaderboard_used_for_selection": False,
                "competition_submission_performed": False,
                "association_configuration": FROZEN_ASSOCIATION_CONFIGURATION,
                "models": models,
                "appearance_models": appearances,
                "appearance_blend": blends,
            }
        ),
        encoding="utf-8",
    )
    return evidence, trackastra_dirs, appearance_models


def test_candidate_builder_requires_exact_appearance_acceptance(tmp_path: Path) -> None:
    evidence, trackastra_dirs, appearance_models = accepted_fixture(tmp_path)

    payload = load_acceptance(evidence, trackastra_dirs, appearance_models)

    assert payload["candidate_family"] == "trackastra_appearance_blend"
    assert payload["appearance_blend"]["target_6bba"]["appearance_weight"] == 0.2


def test_candidate_builder_rejects_appearance_checkpoint_mutation(tmp_path: Path) -> None:
    evidence, trackastra_dirs, appearance_models = accepted_fixture(tmp_path)
    appearance_models["target_44b6"].write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="appearance hash mismatch"):
        load_acceptance(evidence, trackastra_dirs, appearance_models)


def test_candidate_builder_accepts_exact_pair_fusion_evidence(tmp_path: Path) -> None:
    evidence, trackastra_dirs, appearance_models = accepted_fixture(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload.update(
        candidate_family="trackastra_pair_fusion_blend",
        appearance_family="temporal_pair_fusion_v2",
    )
    pair_contract = {
        "appearance_family": "temporal_pair_fusion_v2",
        "parameter_count": 20_869_325,
        "pair_feature_width": 1_029,
        "pair_projection_width": 1_024,
        "pair_hidden_widths": [512, 128],
        "pair_fusion_policy": "candidate-limited source-target-absolute-product-displacement-division MLP",
        "pair_loss_policy": "all-positive candidate-pair mean-log-probability",
        "embedding_auxiliary_loss_weight": 0.25,
        "pair_chunk_size": 4_096,
    }
    for model in payload["appearance_models"].values():
        model.update(pair_contract)
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted = load_acceptance(evidence, trackastra_dirs, appearance_models)

    assert accepted["appearance_family"] == "temporal_pair_fusion_v2"
    payload["appearance_family"] = "temporal_cosine_v1"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="accepted exact candidate"):
        load_acceptance(evidence, trackastra_dirs, appearance_models)


def test_candidate_builder_accepts_exact_contextual_v3_evidence(
    tmp_path: Path,
) -> None:
    evidence, trackastra_dirs, appearance_models = accepted_fixture(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload.update(
        candidate_family="trackastra_contextual_pair_fusion_blend",
        appearance_family="temporal_contextual_pair_fusion_v3",
    )
    contextual_contract = {
        "appearance_family": "temporal_contextual_pair_fusion_v3",
        "parameter_count": 20_747_761,
        "candidate_context_width": 18,
        "contextual_pair_feature_width": 1_047,
        "edge_token_width": 256,
        "edge_set_feature_width": 1_536,
        "edge_head_hidden_widths": [512, 128],
        "contextual_pair_policy": (
            "candidate-limited temporal-context outgoing-incoming edge-set pooling"
        ),
        "transition_context_policy": (
            "bounded phase-correlation with projection refinement, duplicate evidence, "
            "and robust residual-motion context"
        ),
            "pair_loss_policy": (
                "outgoing all-positive child ranking plus eligible incoming "
                "one-parent ranking"
            ),
            "reciprocal_parent_loss_weight": 0.35,
        "embedding_auxiliary_loss_weight": 0.25,
        "pair_chunk_size": 4_096,
    }
    for model in payload["appearance_models"].values():
        model.update(contextual_contract)
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted = load_acceptance(evidence, trackastra_dirs, appearance_models)

    assert accepted["appearance_family"] == "temporal_contextual_pair_fusion_v3"


def test_candidate_builder_accepts_exact_predeclared_trackastra_control(
    tmp_path: Path,
) -> None:
    evidence, trackastra_dirs, appearance_models = accepted_fixture(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    for model in payload["models"].values():
        model.update(
            {"best_step": 0, "source_policy": "predeclared_pretrained_control"}
        )
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted = load_acceptance(evidence, trackastra_dirs, appearance_models)
    assert all(row["best_step"] == 0 for row in accepted["models"].values())


def test_appearance_shard_weight_includes_pair_and_patch_work() -> None:
    video = GraphVideo(
        "44b6_fixture",
        node_ids=np.arange(9),
        times=np.asarray([0, 0, 1, 1, 1, 2, 2, 2, 2]),
        coords_voxel=np.zeros((9, 3), dtype=np.float32),
        edges=np.empty((0, 2), dtype=np.int64),
    )

    assert appearance_movie_inference_weight(video) == (
        2 * 3 + 3 * 4 + APPEARANCE_NODE_COST * 9
    )
    assert appearance_movie_inference_weight(video, encoder_count=2) == (
        2 * 3 + 3 * 4 + 2 * APPEARANCE_NODE_COST * 9
    )
    assert appearance_movie_inference_weight(video, pair_fusion=True) == (
        PAIR_FUSION_PAIR_COST * (2 * 3 + 3 * 4) + APPEARANCE_NODE_COST * 9
    )
