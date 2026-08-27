from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.temporal_contrastive.dual_fold_appearance_submission import (
    APPEARANCE_NODE_COST,
    appearance_movie_inference_weight,
    load_acceptance,
)
from research.trackastra_graph.dual_fold_processed_acceptance import (
    FROZEN_ASSOCIATION_CONFIGURATION,
)
from research.trackastra_graph.train_biohub_graph_transformer import GraphVideo
import numpy as np


FOLDS = ("target_44b6", "target_6bba")


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
            "parameter_count": 19_218_498,
            "checkpoint_weight_source": "optimizer-step exponential moving average",
            "ema_decay": 0.997,
        }
        blends[fold] = {
            "appearance_weight": index * 0.1,
            "division_weight": index * 0.05,
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
