from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/temporal_contrastive/train_graph_context_division_sweep.py"
SPEC = importlib.util.spec_from_file_location("graph_context_trainer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_augmentation_keeps_context_aligned_and_daughter_symmetric() -> None:
    patches = torch.zeros(4, 3, 3, 3, 3, 3)
    geometry = torch.arange(36, dtype=torch.float32).reshape(4, 9)
    context = torch.zeros(4, 43, 8)
    context[:, 0, 5] = 1.0
    context[:, 1:3, 6] = 1.0
    context[:, 3:9, 7] = 1.0
    context[..., 1:4] = torch.tensor((0.25, -0.50, 0.75))
    generator = torch.Generator().manual_seed(19)

    augmented, augmented_geometry, augmented_context = MODULE.augment_batch(
        patches, geometry, context, generator
    )

    assert augmented.shape == patches.shape
    assert augmented_geometry.shape == geometry.shape
    assert augmented_context.shape == context.shape
    torch.testing.assert_close(augmented_context[..., 5:], context[..., 5:])
    assert torch.all(torch.abs(augmented_context[..., 1:4]) == torch.abs(context[..., 1:4]))


def test_balanced_sampler_covers_each_class_geometry_stratum() -> None:
    targets = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    eligible = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    rows = MODULE.balanced_rows(
        targets, eligible, 12, torch.Generator().manual_seed(7)
    )
    observed = {
        (bool(targets[index] > 0.5), bool(eligible[index])) for index in rows
    }

    assert observed == {(True, True), (True, False), (False, True), (False, False)}


def test_trainer_is_large_sequential_and_sealed_audit_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'default="613111,713117,813121,913127"' in source
    assert 'default=20_000' in source
    assert 'len(args.initial_model) != 2 or len(seeds) != 4' in source
    assert '"planned_model_count": 8' in source
    assert '"partial_checkpoint_resumed": False' in source
    assert '"resumed_completed_member_count": len(resumed_members)' in source
    assert '"parameter_count": 74_732_308' not in source
    assert '"audit_opened": bool(accepted)' in source
    assert '"ensemble_members_precommitted_before_audit": True' in source
    assert 'set(precommitted_members) <= independently_strong' in source
    assert '"model_subset_searched_on_audit": False' in source
    assert '"absolute_threshold_used_for_deployment": False' in source
    assert '"final_probe_opened": False' in source
    assert '"public_leaderboard_used_for_selection": False' in source
    assert '"authorized_for_submission": False' in source
    assert "kaggle competitions submit" not in source


def _completed_terminal(member: str, seed: int, model_hash: str, initial_hash: str) -> dict:
    selection = {
        "average_precision": 0.90,
        "true_positives_before_first_false_positive": 3,
        "by_embryo": {
            "44b6": {"average_precision": 0.80},
            "6bba": {"average_precision": 0.85},
        },
    }
    return {
        "schema_version": 1,
        "status": "accepted_at_selection",
        "run_id": MODULE.RUN_ID,
        "member": member,
        "seed": seed,
        "completed_steps": 20_000,
        "best_step": 250,
        "selection": selection,
        "selection_frozen_threshold": {"fp": 0, "tp": 3},
        "selection_gate_passed": True,
        "model_sha256": model_hash,
        "initial_backbone_sha256": initial_hash,
        **MODULE.architecture_contract(),
        "audit_opened": False,
        "final_probe_opened": False,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_audit": True,
        "authorized_for_submission": False,
    }


def test_completed_member_resume_is_hash_bound_and_pre_audit(tmp_path: Path) -> None:
    member = "seed-613111-init-1"
    member_root = tmp_path / member
    member_root.mkdir()
    checkpoint = member_root / "graph_context_model.pt"
    checkpoint.write_bytes(b"owned checkpoint")
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"owned initialization")
    checkpoint_hash = MODULE.sha256_file(checkpoint)
    initial_hash = MODULE.sha256_file(initial)
    terminal = _completed_terminal(member, 613111, checkpoint_hash, initial_hash)
    (member_root / "selection_history.json").write_text(
        json.dumps({"rows": [{"step": 250}]}), encoding="utf-8"
    )
    (member_root / "worker_terminal.json").write_text(
        json.dumps(terminal), encoding="utf-8"
    )

    resumed = MODULE.load_completed_member(
        member_root=member_root,
        member_name=member,
        seed=613111,
        initial_model_path=initial,
        steps=20_000,
    )

    assert resumed == terminal


def test_completed_member_resume_rejects_drift_and_opened_audit(tmp_path: Path) -> None:
    member = "seed-613111-init-1"
    member_root = tmp_path / member
    member_root.mkdir()
    checkpoint = member_root / "graph_context_model.pt"
    checkpoint.write_bytes(b"owned checkpoint")
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"owned initialization")
    terminal = _completed_terminal(
        member, 613111, MODULE.sha256_file(checkpoint), MODULE.sha256_file(initial)
    )
    (member_root / "selection_history.json").write_text(
        json.dumps({"rows": [{"step": 250}]}), encoding="utf-8"
    )
    (member_root / "worker_terminal.json").write_text(
        json.dumps(terminal), encoding="utf-8"
    )
    checkpoint.write_bytes(b"changed checkpoint")

    try:
        MODULE.load_completed_member(
            member_root=member_root,
            member_name=member,
            seed=613111,
            initial_model_path=initial,
            steps=20_000,
        )
    except ValueError as error:
        assert "integrity checks" in str(error)
    else:
        raise AssertionError("checkpoint drift was accepted")

    checkpoint.write_bytes(b"owned checkpoint")
    (member_root / "audit_terminal.json").write_text("{}", encoding="utf-8")
    try:
        MODULE.load_completed_member(
            member_root=member_root,
            member_name=member,
            seed=613111,
            initial_model_path=initial,
            steps=20_000,
        )
    except ValueError as error:
        assert "already opened audit" in str(error)
    else:
        raise AssertionError("opened audit state was accepted")
