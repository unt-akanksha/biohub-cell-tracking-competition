from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.temporal_contrastive import train_dual_fold_pair_fusion as generic
from research.temporal_contrastive.contextual_pair_fusion import (
    ContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
    CONTEXTUAL_PAIR_FUSION_FAMILY,
    discover_shards,
    inventory_sha256,
    load_shard,
    shard_forward,
)


def write_shard(root: Path, source: str, role: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{source}-t0100.npz"
    rng = np.random.default_rng(13)
    candidates = np.asarray(
        [[True, True, False], [False, True, True]], dtype=bool
    )
    positives = np.asarray(
        [[True, False, False], [False, False, True]], dtype=bool
    )
    np.savez_compressed(
        path,
        source_patches=rng.normal(size=(2, 3, 17, 17, 17)).astype(np.float16),
        target_patches=rng.normal(size=(3, 3, 17, 17, 17)).astype(np.float16),
        source_coords_um=np.asarray([[0, 0, 0], [0, 8, 0]], dtype=np.float32),
        target_coords_um=np.asarray(
            [[0, 1, 0], [0, 7, 0], [0, 9, 0]], dtype=np.float32
        ),
        candidate_mask=candidates,
        positive_mask=positives,
        division_target=np.zeros(2, dtype=np.float32),
        candidate_context=np.zeros((2, 3, 18), dtype=np.float32),
    )
    shard_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": source,
        "source_role": role,
        "csv_timepoint": 100,
        "organizer_declared_test_overlap": False,
        "competition_test_data_read": False,
        "public_competition_predictions_read": False,
        "leaderboard_used": False,
        "submission_created": False,
        "candidate_context_width": 18,
        "shard": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": shard_hash,
        },
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def test_pretraining_shard_discovery_is_role_and_hash_bound(tmp_path: Path) -> None:
    path = write_shard(tmp_path, "ZSNS004", "external_pretraining")
    records = discover_shards(
        tmp_path,
        expected_source="ZSNS004",
        expected_role="external_pretraining",
    )

    assert [row.path for row in records] == [path]
    assert len(inventory_sha256(records)) == 64
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["leaderboard_used"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid ZebraHub shard evidence"):
        discover_shards(
            tmp_path,
            expected_source="ZSNS004",
            expected_role="external_pretraining",
        )


def test_pretraining_shard_runs_contextual_loss_backward(tmp_path: Path) -> None:
    path = write_shard(tmp_path, "ZSNS004", "external_pretraining")
    model = ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    batch = load_shard(path, torch.device("cpu"))

    logits, pair_loss, embedding_loss, division_loss = shard_forward(
        model, batch, patch_batch_size=2
    )
    loss = pair_loss + 0.25 * embedding_loss + 0.20 * division_loss
    loss.backward()

    assert torch.isfinite(logits[batch["candidate_mask"]]).all()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_finetuning_initialization_requires_complete_hash_bound_evidence(
    tmp_path: Path,
) -> None:
    fold = "target_44b6"
    fold_dir = tmp_path / fold
    fold_dir.mkdir(parents=True)
    model = torch.nn.Linear(1, 1)
    model_path = fold_dir / "pretrained_model.pt"
    torch.save(model.state_dict(), model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    worker = {
        "status": "completed",
        "run_id": "zebrahub-contextual-pretrain-v1",
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "fold": fold,
        "parameter_count": 2,
        "best_step": 1,
        "model_sha256": model_hash,
        "external_training_source": "ZSNS004",
        "external_validation_source": "ZSNS005",
        "competition_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    (fold_dir / "worker_terminal.json").write_text(
        json.dumps(worker), encoding="utf-8"
    )
    aggregate = {
        "status": "completed",
        "run_id": "zebrahub-contextual-pretrain-v1",
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "gpu_count": 2,
        "both_folds_improved": True,
        "submission_created": False,
    }
    (tmp_path / "pretraining_terminal.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    original_count = generic.EXPECTED_PARAMETER_COUNT
    try:
        generic.EXPECTED_PARAMETER_COUNT = 2
        evidence = generic.load_initial_model(
            torch.nn.Linear(1, 1), tmp_path, fold
        )
        assert evidence["model_sha256"] == model_hash
        worker["external_validation_source"] = "ZSNS004"
        (fold_dir / "worker_terminal.json").write_text(
            json.dumps(worker), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="invalid external pretraining"):
            generic.load_initial_model(torch.nn.Linear(1, 1), tmp_path, fold)
    finally:
        generic.EXPECTED_PARAMETER_COUNT = original_count
