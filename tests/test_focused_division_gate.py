from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import research.temporal_contrastive.train_focused_division_gate as focused


def test_classification_metrics_reward_clean_division_ranking() -> None:
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    clean = focused.classification_metrics(
        torch.tensor([4.0, -3.0, 2.0, -2.0]), targets
    )
    reversed_order = focused.classification_metrics(
        torch.tensor([-4.0, 3.0, -2.0, 2.0]), targets
    )

    assert clean["average_precision"] == pytest.approx(1.0)
    assert clean["best_jaccard"] == pytest.approx(1.0)
    assert clean["recall_at_precision_0_80"] == pytest.approx(1.0)
    assert clean["average_precision"] > reversed_order["average_precision"]


def test_focused_gate_requires_ap_and_jaccard_gain_without_precision_recall_loss() -> None:
    baseline = {
        "rows": 100,
        "positives": 10,
        "average_precision": 0.50,
        "best_jaccard": 0.40,
        "recall_at_precision_0_80": 0.20,
    }
    accepted = {
        **baseline,
        "average_precision": 0.53,
        "best_jaccard": 0.45,
        "recall_at_precision_0_80": 0.30,
    }
    precision_regression = {**accepted, "recall_at_precision_0_80": 0.10}

    assert focused.focused_selection_gate(baseline, accepted)["passed"] is True
    assert focused.focused_selection_gate(baseline, precision_regression)["passed"] is False


def test_balanced_rows_contains_both_classes() -> None:
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    generator = torch.Generator().manual_seed(12)

    rows = focused.balanced_rows(targets, 9, generator)

    assert rows.shape == (9,)
    assert int((targets[rows] > 0.5).sum()) == 3
    assert int((targets[rows] < 0.5).sum()) == 6


def test_rejected_v4_is_accepted_only_as_prediction_preserving_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = "target_44b6"
    initialization = {
        "source_family": "temporal_contextual_pair_fusion_v3",
        "target_family": focused.MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "initial_predictions_numerically_preserved": True,
    }
    selection = {"composite": 0.9}
    audit = {"composite": 0.8}
    worker = {
        "status": "completed",
        "fold": fold,
        "parameter_count": focused.EXPECTED_PARAMETER_COUNT,
        "best_step": 0,
        "selection_gate_passed": False,
        "audit_gate_passed": False,
        "best_selection": selection,
        "initial_selection": selection,
        "final_audit": audit,
        "initial_audit": audit,
        "model_sha256": "modelhash",
        "initialization": initialization,
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": focused.PARENT_RUN_ID,
        "appearance_family": focused.MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "gpu_count": 2,
        "both_folds_improved": False,
        "folds": {fold: worker},
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    fold_root = tmp_path / fold
    fold_root.mkdir()
    (tmp_path / "pretraining_terminal.json").write_text(json.dumps(aggregate))
    (fold_root / "worker_terminal.json").write_text(json.dumps(worker))
    (fold_root / "pretrained_model.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(
        focused,
        "sha256_file",
        lambda path: "modelhash" if path.name == "pretrained_model.pt" else "workerhash",
    )
    monkeypatch.setattr(focused.torch, "load", lambda *args, **kwargs: {})

    evidence = focused.load_rejected_zero_residual_parent(
        torch.nn.Identity(), tmp_path, fold
    )

    assert evidence["parent_association_gate_passed"] is False
    assert evidence["authorized_for_focused_division_initialization_only"] is True


def test_rejected_v4_bootstrap_refuses_changed_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = "target_44b6"
    worker = {
        "status": "completed",
        "fold": fold,
        "parameter_count": focused.EXPECTED_PARAMETER_COUNT,
        "best_step": 1,
        "selection_gate_passed": False,
        "audit_gate_passed": False,
        "model_sha256": "modelhash",
        "initialization": {},
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": focused.PARENT_RUN_ID,
        "appearance_family": focused.MULTISCALE_CONTEXTUAL_PAIR_FUSION_FAMILY,
        "gpu_count": 2,
        "both_folds_improved": False,
        "folds": {fold: worker},
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    fold_root = tmp_path / fold
    fold_root.mkdir()
    (tmp_path / "pretraining_terminal.json").write_text(json.dumps(aggregate))
    (fold_root / "worker_terminal.json").write_text(json.dumps(worker))
    (fold_root / "pretrained_model.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(focused, "sha256_file", lambda _path: "modelhash")

    with pytest.raises(ValueError, match="ineligible zero-residual"):
        focused.load_rejected_zero_residual_parent(torch.nn.Identity(), tmp_path, fold)
