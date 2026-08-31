from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify-antelume-graph-context-division-harvest.py"
SPEC = importlib.util.spec_from_file_location("graph_context_harvest", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def build_harvest(path: Path, *, unbound: bool = False, recovered: bool = False) -> None:
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": VERIFIER.RUN_ID,
        "family": VERIFIER.FAMILY,
        "parameter_count": VERIFIER.EXPECTED_PARAMETER_COUNT,
        "planned_model_count": 8,
        "completed_model_count": 8,
        "steps_per_model": 20_000,
        "ensemble_members_precommitted_before_audit": True,
        "precommitted_members": ["seed-1-init-1"],
        "policy_audit_passed": True,
        "deployment_policy": "strongest_selection_individual",
        "deployment_members": ["seed-1-init-1"],
        "absolute_threshold_used_for_deployment": False,
        "model_subset_searched_on_audit": False,
        "selection_accepted_members": ["seed-1-init-1"],
        "independently_strong_members": ["seed-1-init-1"],
        "ensemble_eligible": False,
        "final_probe_opened": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "authorized_for_submission": False,
    }
    probe = {
        "schema_version": 1,
        "status": "development_probe_complete",
        "run_id": VERIFIER.PROBE_RUN_ID,
        "selection_policy": "strongest_selection_individual",
        "member_count": 1,
        "absolute_threshold_used": False,
        "weights_searched_on_probe": False,
        "model_subset_searched_on_probe": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "authorized_for_submission": False,
    }
    files = {
        f"{VERIFIER.ROOT}/training.exit-code": b"0\n",
        f"{VERIFIER.ROOT}/development-probe.exit-code": b"0\n",
        f"{VERIFIER.ROOT}/training.log": b"complete\n",
        f"{VERIFIER.ROOT}/models/graph_context_division_sweep_terminal.json": json.dumps(aggregate).encode(),
        f"{VERIFIER.ROOT}/graph_context_development_probe.json": json.dumps(probe).encode(),
    }
    if recovered:
        aggregate["resumed_completed_member_count"] = len(VERIFIER.RECOVERED_MEMBERS)
        aggregate["resumed_completed_members"] = sorted(VERIFIER.RECOVERED_MEMBERS)
        aggregate["partial_checkpoint_resumed"] = False
        files[f"{VERIFIER.ROOT}/models/graph_context_division_sweep_terminal.json"] = json.dumps(aggregate).encode()
        recovery = {
            "schema_version": 1,
            "run_id": VERIFIER.RECOVERY_RUN_ID,
            "status": "five_completed_members_recovered",
            "recovered_member_count": len(VERIFIER.RECOVERED_MEMBERS),
            "recovered_members": [
                {"member": member, "model_sha256": "0" * 64}
                for member in sorted(VERIFIER.RECOVERED_MEMBERS)
            ],
            "partial_checkpoint_resumed": False,
            "audit_opened_before_recovery": False,
            "trainer_sha256": VERIFIER.RECOVERY_TRAINER_SHA256,
            "competition_test_data_read": False,
            "public_leaderboard_used_for_selection": False,
            "authorized_for_submission": False,
        }
        files[f"{VERIFIER.ROOT}/recovery-manifest.json"] = json.dumps(recovery).encode()
    files[f"{VERIFIER.ROOT}/SHA256SUMS"] = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n"
        for name, value in sorted(files.items())
    ).encode()
    if unbound:
        files[f"{VERIFIER.ROOT}/unbound.txt"] = b"no"
    with tarfile.open(path, "w:gz") as archive:
        for name, value in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))


def test_harvest_recovers_sealed_policy_and_probe(tmp_path: Path) -> None:
    archive = tmp_path / "result.tar.gz"
    build_harvest(archive)

    result = VERIFIER.verify_archive(archive)

    assert result["status"] == "harvest_verified"
    assert result["training_exit_code"] == 0
    assert result["development_probe_exit_code"] == 0
    assert result["completed_model_count"] == 8
    assert result["deployment_members"] == ["seed-1-init-1"]
    assert result["development_probe_present"] is True
    assert result["authorized_for_submission"] is False


def test_harvest_rejects_unbound_files(tmp_path: Path) -> None:
    archive = tmp_path / "result.tar.gz"
    build_harvest(archive, unbound=True)

    with pytest.raises(ValueError, match="unbound"):
        VERIFIER.verify_archive(archive)


def test_harvest_binds_five_completed_recovery_members(tmp_path: Path) -> None:
    archive = tmp_path / "recovered.tar.gz"
    build_harvest(archive, recovered=True)

    result = VERIFIER.verify_archive(archive)

    assert result["resumed_completed_member_count"] == 5
    assert result["recovery_manifest_present"] is True


def test_harvester_uses_one_keepalive_session_and_never_submits() -> None:
    source = (
        ROOT / "scripts/wait-harvest-antelume-graph-context-division-sweep.ps1"
    ).read_text(encoding="utf-8")

    assert "deployed_and_queued" in source
    assert "ServerAliveInterval=60" in source
    assert 'cat "$archive"' in source
    assert "Start-Process -FilePath ssh.exe" in source
    assert "verify-antelume-graph-context-division-harvest.py" in source
    assert "kaggle competitions submit" not in source
    assert 'competition_submission_performed"] = $false' in source
