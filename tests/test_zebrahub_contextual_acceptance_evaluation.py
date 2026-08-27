from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from research.temporal_contrastive import (
    evaluate_zebrahub_contextual_acceptance as acceptance,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pretraining_source(root: Path) -> None:
    folds: dict[str, object] = {}
    for fold in acceptance.FOLDS:
        fold_root = root / fold
        fold_root.mkdir(parents=True)
        model_path = fold_root / "pretrained_model.pt"
        torch.save({"placeholder": torch.tensor([1.0])}, model_path)
        row = {
            "schema_version": 1,
            "status": "completed",
            "run_id": acceptance.PRETRAINING_RUN_ID,
            "appearance_family": acceptance.CONTEXTUAL_PAIR_FUSION_FAMILY,
            "fold": fold,
            "selection_gate_passed": True,
            "audit_gate_passed": True,
            "best_step": 500,
            "validation_partition_policy": acceptance.VALIDATION_PARTITION_POLICY,
            "dataset_manifest_sha256": (
                acceptance.EXPECTED_PRETRAINING_DATASET_MANIFEST_SHA256
            ),
            "parameter_count": acceptance.EXPECTED_PARAMETER_COUNT,
            "reciprocal_parent_loss_weight": (
                acceptance.RECIPROCAL_PARENT_LOSS_WEIGHT
            ),
            "competition_data_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "model_sha256": sha256_file(model_path),
        }
        (fold_root / "worker_terminal.json").write_text(
            json.dumps(row), encoding="utf-8"
        )
        (fold_root / "training_config.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "seed": acceptance.FOLD_SEEDS[fold],
                    "dataset_manifest_sha256": (
                        acceptance.EXPECTED_PRETRAINING_DATASET_MANIFEST_SHA256
                    ),
                }
            ),
            encoding="utf-8",
        )
        folds[fold] = row
    (root / "pretraining_terminal.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "run_id": acceptance.PRETRAINING_RUN_ID,
                "gpu_count": 2,
                "both_folds_improved": True,
                "competition_data_read": False,
                "public_predictions_copied": False,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )


def test_pretraining_source_requires_two_frozen_passing_folds(
    tmp_path: Path,
) -> None:
    write_pretraining_source(tmp_path)

    evidence = acceptance.verify_pretraining_source(tmp_path)

    assert set(evidence["folds"]) == set(acceptance.FOLDS)
    assert all(
        len(evidence["folds"][fold]["model_sha256"]) == 64
        for fold in acceptance.FOLDS
    )
    terminal_path = tmp_path / "pretraining_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["folds"][acceptance.FOLDS[0]]["audit_gate_passed"] = False
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="worker/aggregate divergence"):
        acceptance.verify_pretraining_source(tmp_path)


def test_initialization_is_repeatable_without_opening_acceptance_data() -> None:
    device = torch.device("cpu")
    acceptance.seed_initialization(51_004, device)
    first = acceptance.ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    first_hash = acceptance.state_dict_sha256(first)

    acceptance.seed_initialization(51_004, device)
    second = acceptance.ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )
    acceptance.seed_initialization(61_007, device)
    third = acceptance.ContextualPairFusionAssociationModel(
        base_channels=8, embedding_channels=16
    )

    assert acceptance.state_dict_sha256(second) == first_hash
    assert acceptance.state_dict_sha256(third) != first_hash


def test_acceptance_source_binds_published_and_record_inventories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = {
        "status": "verified_unopened",
        "manifest_sha256": acceptance.EXPECTED_ACCEPTANCE_MANIFEST_SHA256,
        "inventory_sha256": acceptance.EXPECTED_ACCEPTANCE_INVENTORY_SHA256,
        "shards": acceptance.ACCEPTANCE_SHARDS,
        "model_predictions_read": False,
    }
    records = [object() for _ in range(acceptance.ACCEPTANCE_SHARDS)]
    monkeypatch.setattr(acceptance, "verify_acceptance", lambda _root: evidence)
    monkeypatch.setattr(
        acceptance,
        "discover_shards",
        lambda *_args, **_kwargs: records,
    )
    monkeypatch.setattr(
        acceptance,
        "inventory_sha256",
        lambda _records: acceptance.EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256,
    )

    verified, discovered = acceptance.verify_acceptance_source(tmp_path)

    assert verified == evidence
    assert discovered == records
    assert (
        acceptance.EXPECTED_ACCEPTANCE_INVENTORY_SHA256
        != acceptance.EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256
    )
    monkeypatch.setattr(acceptance, "inventory_sha256", lambda _records: "0" * 64)
    with pytest.raises(ValueError, match="inventory changed"):
        acceptance.verify_acceptance_source(tmp_path)


def test_acceptance_evaluator_is_two_gpu_one_shot_and_non_submission() -> None:
    source = Path(acceptance.__file__).read_text(encoding="utf-8").casefold()

    assert "torch.cuda.device_count() != 2" in source
    assert 'environment["cuda_visible_devices"] = tokens[index]' in source
    assert "one-shot acceptance output directory is not empty" in source
    assert "selection_or_checkpoint_redirect_permitted\": false" in source
    assert "public_predictions_copied\": false" in source
    assert "submission_created\": false" in source
    assert "kaggle competitions submit" not in source
