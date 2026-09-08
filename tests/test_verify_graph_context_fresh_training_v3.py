from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify-graph-context-fresh-training-v3.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_fresh_graph_v3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def rejected_fixture(root: Path, module) -> None:
    for member, (seed, fold) in module.EXPECTED_MEMBERS.items():
        member_root = root / "results" / member
        member_root.mkdir(parents=True)
        checkpoint = member_root / "graph_context_model.pt"
        checkpoint.write_bytes(member.encode())
        (member_root / "selection_history.json").write_text("{}", encoding="utf-8")
        write_json(
            member_root / "worker_terminal.json",
            {
                "run_id": module.RUN_ID,
                "member": member,
                "seed": seed,
                "warm_start_fold": fold,
                "fresh_split_sha256": module.SPLIT_SHA256,
                "completed_steps": 20_000,
                "parameter_count": module.PARAMETER_COUNT,
                "model_sha256": hashlib.sha256(member.encode()).hexdigest(),
                "audit_opened": False,
                "final_probe_opened": False,
                "competition_test_data_read": False,
                "public_code_copied": False,
                "public_predictions_copied": False,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
                "authorized_for_submission": False,
                "selection": {"average_precision": 0.1},
                "selection_gate_passed": False,
            },
        )
    write_json(
        root / "results/graph_context_fresh_ensemble_terminal.json",
        {
            "schema_version": 1,
            "status": "rejected_at_selection",
            "run_id": module.RUN_ID,
            "reason": "fewer_than_two_selection_admitted_members",
            "fresh_split_sha256": module.SPLIT_SHA256,
            "runtime_manifest_sha256": module.RUNTIME_MANIFEST_SHA256,
            "required_visible_gpu_count": 2,
            "completed_model_count": 4,
            "competition_test_data_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_full_candidate_evaluation": False,
            "authorized_for_submission": False,
        },
    )


def test_verified_rejection_cannot_authorize_candidate(tmp_path: Path) -> None:
    module = load_module()
    rejected_fixture(tmp_path, module)
    report = module.verify(tmp_path)
    assert report["status"] == "verified_rejection"
    assert report["authorized_for_full_candidate_evaluation"] is False
    assert report["authorized_for_submission"] is False


def test_member_hash_tamper_fails_closed(tmp_path: Path) -> None:
    module = load_module()
    rejected_fixture(tmp_path, module)
    checkpoint = (
        tmp_path / "results/seed-1409101-init-1/graph_context_model.pt"
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="member evidence failed"):
        module.verify(tmp_path)
