from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.temporal_contrastive import (
    verify_zebrahub_contextual_acceptance_output as verifier,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def metrics(offset: float) -> dict:
    return {
        "composite": 0.20 + offset,
        "top1": 0.25 + offset,
        "mrr": 0.30 + offset,
        "division_top2": 0.35 + offset,
        "rows": 100,
        "division_rows": 10,
        "transitions": 16,
    }


def build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    output = tmp_path / "output"
    pretraining = tmp_path / "pretraining"
    pretraining.mkdir()
    source_folds = {}
    for index, fold in enumerate(verifier.FOLDS):
        model = pretraining / fold / "pretrained_model.pt"
        model.parent.mkdir()
        model.write_bytes(f"model-{fold}".encode())
        source_folds[fold] = {
            "model_path": model,
            "model_sha256": f"{index + 1}" * 64,
            "worker_terminal_sha256": f"{index + 3}" * 64,
        }
    monkeypatch.setattr(
        verifier,
        "verify_pretraining_source",
        lambda _root: {
            "terminal_sha256": "a" * 64,
            "folds": source_folds,
        },
    )
    monkeypatch.setattr(verifier, "strict_load_checkpoint", lambda _path: None)
    monkeypatch.setattr(
        verifier, "initial_state_hash", lambda seed: f"initial-{seed}"
    )

    aggregate_folds = {}
    aggregate_root = output / "zebrahub_contextual_acceptance_v1"
    for fold in verifier.FOLDS:
        initial = metrics(0.0)
        final = metrics(0.10)
        terminal = {
            "schema_version": 1,
            "status": "completed",
            "run_id": verifier.RUN_ID,
            "fold": fold,
            "seed": verifier.FOLD_SEEDS[fold],
            "elapsed_seconds": 100.0,
            "appearance_family": verifier.CONTEXTUAL_PAIR_FUSION_FAMILY,
            "parameter_count": verifier.EXPECTED_PARAMETER_COUNT,
            "acceptance_source": verifier.ACCEPTANCE_SOURCE,
            "acceptance_role": verifier.ACCEPTANCE_ROLE,
            "acceptance_shards": verifier.ACCEPTANCE_SHARDS,
            "acceptance_cache_bytes": 12345,
            "acceptance_manifest_sha256": verifier.EXPECTED_ACCEPTANCE_MANIFEST_SHA256,
            "acceptance_inventory_sha256": verifier.EXPECTED_ACCEPTANCE_INVENTORY_SHA256,
            "acceptance_record_inventory_sha256": verifier.EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256,
            "pretraining_terminal_sha256": "a" * 64,
            "pretrained_model_sha256": source_folds[fold]["model_sha256"],
            "pretraining_worker_terminal_sha256": source_folds[fold][
                "worker_terminal_sha256"
            ],
            "initial_state_sha256": f"initial-{verifier.FOLD_SEEDS[fold]}",
            "initial": initial,
            "final": final,
            "gate": verifier.validation_improvement_gate(initial, final),
            "gate_passed": True,
            "selection_or_checkpoint_redirect_permitted": False,
            "competition_data_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        }
        terminal_path = aggregate_root / fold / "acceptance_terminal.json"
        write_json(terminal_path, terminal)
        aggregate_folds[fold] = {
            **terminal,
            "terminal_sha256": verifier.sha256_file(terminal_path),
        }
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": verifier.RUN_ID,
        "elapsed_seconds": 120.0,
        "gpu_count": 2,
        "pretraining_run_id": verifier.PRETRAINING_RUN_ID,
        "pretraining_terminal_sha256": "a" * 64,
        "acceptance_source": verifier.ACCEPTANCE_SOURCE,
        "acceptance_manifest_sha256": verifier.EXPECTED_ACCEPTANCE_MANIFEST_SHA256,
        "acceptance_inventory_sha256": verifier.EXPECTED_ACCEPTANCE_INVENTORY_SHA256,
        "acceptance_record_inventory_sha256": verifier.EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256,
        "both_folds_improved": True,
        "folds": aggregate_folds,
        "selection_or_checkpoint_redirect_permitted": False,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    aggregate_path = aggregate_root / "acceptance_terminal.json"
    write_json(aggregate_path, aggregate)
    launcher = {
        "schema_version": 1,
        "run_id": verifier.RUN_ID,
        "status": "completed",
        "elapsed_seconds": 125.0,
        "declared_budget_seconds": 3600,
        "evaluator_hard_stop_seconds": 3300,
        "acceptance_terminal_exists": True,
        "acceptance_terminal_sha256": verifier.sha256_file(aggregate_path),
        "gpu_count_required": 2,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    write_json(output / "launcher_terminal.json", launcher)
    return output, pretraining


def test_downloaded_acceptance_output_is_hash_bound_and_gate_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, pretraining = build_fixture(tmp_path, monkeypatch)

    result = verifier.verify_acceptance_output(output, pretraining)

    assert result["status"] == "accepted_verified"
    assert result["both_folds_improved"] is True
    assert set(result["folds"]) == set(verifier.FOLDS)
    assert result["submission_created"] is False


def test_downloaded_acceptance_rejects_aggregate_fold_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, pretraining = build_fixture(tmp_path, monkeypatch)
    aggregate_path = (
        output / "zebrahub_contextual_acceptance_v1" / "acceptance_terminal.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["folds"][verifier.FOLDS[0]]["final"]["top1"] += 0.01
    write_json(aggregate_path, aggregate)
    launcher_path = output / "launcher_terminal.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    launcher["acceptance_terminal_sha256"] = verifier.sha256_file(aggregate_path)
    write_json(launcher_path, launcher)

    with pytest.raises(ValueError, match="diverges from aggregate"):
        verifier.verify_acceptance_output(output, pretraining)


def test_downloaded_acceptance_rejects_submission_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, pretraining = build_fixture(tmp_path, monkeypatch)
    (output / "submission.csv").write_text("id,parent_id\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prohibited artifacts"):
        verifier.verify_acceptance_output(output, pretraining)
