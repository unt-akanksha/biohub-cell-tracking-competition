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
    contextual_bidirectional_pair_nll,
    masked_reciprocal_parent_nll,
)
from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
    CONTEXTUAL_PAIR_FUSION_FAMILY,
    VALIDATION_AUDIT_TIMEPOINTS,
    VALIDATION_PARTITION_POLICY,
    VALIDATION_SELECTION_TIMEPOINTS,
    ShardRecord,
    augment_normalized_patches,
    cached_batch,
    discover_shards,
    inventory_sha256,
    load_shard,
    partition_validation_records,
    preload_shards,
    shard_forward,
    validation_improvement_gate,
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

    logits, pair_loss, parent_loss, embedding_loss, division_loss = shard_forward(
        model, batch, patch_batch_size=2
    )
    loss = pair_loss + 0.35 * parent_loss + 0.25 * embedding_loss + 0.20 * division_loss
    loss.backward()

    assert torch.isfinite(logits[batch["candidate_mask"]]).all()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_reciprocal_parent_loss_skips_uncontested_targets_and_backpropagates() -> None:
    logits = torch.tensor(
        [[3.0, 1.0, float("-inf")], [2.0, float("-inf"), 4.0]],
        requires_grad=True,
    )
    candidates = torch.isfinite(logits)
    positives = torch.tensor(
        [[True, False, False], [False, False, True]], dtype=torch.bool
    )

    parent_loss = masked_reciprocal_parent_nll(logits, positives, candidates)
    combined = contextual_bidirectional_pair_nll(logits, positives, candidates)
    combined.backward()

    assert torch.isfinite(parent_loss)
    assert float(parent_loss.detach()) > 0.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad[candidates]).all()

    uncontested_logits = torch.tensor(
        [[1.0, float("-inf")], [float("-inf"), 2.0]], requires_grad=True
    )
    uncontested_candidates = torch.isfinite(uncontested_logits)
    uncontested = masked_reciprocal_parent_nll(
        uncontested_logits,
        uncontested_candidates,
        uncontested_candidates,
    )
    uncontested.backward()
    assert float(uncontested.detach()) == 0.0
    assert torch.equal(
        uncontested_logits.grad[uncontested_candidates], torch.zeros(2)
    )


def test_preloaded_shards_are_complete_and_assignment_isolated(tmp_path: Path) -> None:
    path = write_shard(tmp_path, "ZSNS004", "external_pretraining")
    records = discover_shards(
        tmp_path,
        expected_source="ZSNS004",
        expected_role="external_pretraining",
    )
    cache, tensor_bytes = preload_shards(records, torch.device("cpu"))
    first = cached_batch(cache, path)
    second = cached_batch(cache, path)
    first["source_patches"] = torch.zeros_like(first["source_patches"])

    assert tensor_bytes > 0
    assert first is not second
    assert not torch.equal(first["source_patches"], second["source_patches"])


def test_zsns005_selection_and_audit_use_disjoint_developmental_windows() -> None:
    timepoints = sorted(
        VALIDATION_SELECTION_TIMEPOINTS | VALIDATION_AUDIT_TIMEPOINTS
    )
    records = [
        ShardRecord(
            path=Path(f"ZSNS005-t{timepoint:04d}.npz"),
            manifest_path=Path(f"ZSNS005-t{timepoint:04d}.manifest.json"),
            sha256=f"{timepoint:064x}",
            manifest_sha256=f"{timepoint + 1:064x}",
            csv_timepoint=timepoint,
        )
        for timepoint in reversed(timepoints)
    ]

    selection, audit = partition_validation_records(records)

    assert {row.csv_timepoint for row in selection} == set(
        VALIDATION_SELECTION_TIMEPOINTS
    )
    assert {row.csv_timepoint for row in audit} == set(
        VALIDATION_AUDIT_TIMEPOINTS
    )
    assert {row.sha256 for row in selection}.isdisjoint(
        {row.sha256 for row in audit}
    )
    with pytest.raises(ValueError, match="timepoint inventory changed"):
        partition_validation_records(records[:-1])


def test_validation_gate_requires_broad_gain_and_unchanged_inventory() -> None:
    initial = {
        "composite": 0.20,
        "top1": 0.15,
        "mrr": 0.25,
        "division_top2": 0.10,
        "rows": 500,
        "division_rows": 50,
        "transitions": 8,
    }
    candidate = {
        **initial,
        "composite": 0.35,
        "top1": 0.30,
        "mrr": 0.40,
        "division_top2": 0.20,
    }

    assert validation_improvement_gate(initial, candidate)["passed"] is True
    assert validation_improvement_gate(
        initial, {**candidate, "division_top2": 0.09}
    )["passed"] is False
    assert validation_improvement_gate(
        initial, {**candidate, "composite": 0.205}
    )["passed"] is False
    assert validation_improvement_gate(
        initial, {**candidate, "rows": 499}
    )["passed"] is False


def test_microscopy_augmentation_is_seeded_finite_and_fixed_shape() -> None:
    patches = torch.linspace(
        -3.0, 3.0, 2 * 3 * 17 * 17 * 17, dtype=torch.float32
    ).reshape(2, 3, 17, 17, 17)
    first_generator = torch.Generator().manual_seed(9107)
    second_generator = torch.Generator().manual_seed(9107)
    first = augment_normalized_patches(
        patches, generator=first_generator, spatial_code=23
    )
    second = augment_normalized_patches(
        patches, generator=second_generator, spatial_code=23
    )

    assert first.shape == patches.shape
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert not torch.equal(first, patches)
    assert float(first.min()) >= -6.0
    assert float(first.max()) <= 6.0


def test_microscopy_augmentation_refuses_changed_patch_contract() -> None:
    generator = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="shape"):
        augment_normalized_patches(
            torch.zeros(1, 3, 15, 17, 17),
            generator=generator,
            spatial_code=0,
        )
    with pytest.raises(ValueError, match="spatial augmentation code"):
        augment_normalized_patches(
            torch.zeros(1, 3, 17, 17, 17),
            generator=generator,
            spatial_code=32,
        )


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
        "selection_gate_passed": True,
        "audit_gate_passed": True,
        "validation_partition_policy": VALIDATION_PARTITION_POLICY,
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
