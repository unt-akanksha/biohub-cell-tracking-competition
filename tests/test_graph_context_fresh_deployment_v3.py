from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-graph-context-fresh-deployment-v3.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fresh_deployment_v3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_runtime(root: Path, module) -> Path:
    root.mkdir()
    for name, source in module.RUNTIME_FILES.items():
        shutil.copy2(source, root / name)
    for name in (
        "fresh_training_terminal.json",
        "fresh_selection_policy.json",
        "fresh_verification_report.json",
    ):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "morphology_model.joblib").write_bytes(b"morphology")
    (root / module.SKLEARN_WHEEL).write_bytes(b"wheel")
    members = []
    for index in range(2):
        name = f"graph_context_model_{index:02d}.pt"
        (root / name).write_bytes(f"model-{index}".encode())
        members.append(
            {
                "path": name,
                "parameter_count": 74_732_308,
                "selection_gate_passed": True,
                "model_sha256": module.sha256_file(root / name),
            }
        )
    policy = {
        "schema_version": 1,
        "status": "fresh_audit_accepted",
        "run_id": module.POLICY_RUN_ID,
        "graph_context_policy": "all-selection-admitted-equal-rank-ensemble",
        "graph_context_member_count": 2,
        "graph_context_members": members,
        "fresh_audit_gate_passed": True,
        "selection_policy_frozen_before_audit": True,
        "maximum_added_edges_per_movie": 1,
        "biological_geometry_minimum": 3.0,
        "absolute_threshold_used": False,
        "authorized_for_full_candidate_evaluation": True,
        "authorized_for_submission": False,
        "morphology_model_sha256": module.sha256_file(
            root / "morphology_model.joblib"
        ),
    }
    module.write_json(root / "graph-context-fresh-policy.json", policy)
    names = {
        *module.RUNTIME_FILES,
        *(row["path"] for row in members),
        "fresh_training_terminal.json",
        "fresh_selection_policy.json",
        "fresh_verification_report.json",
        "morphology_model.joblib",
        "graph-context-fresh-policy.json",
        module.SKLEARN_WHEEL,
    }
    module.write_json(
        root / "GRAPH_CONTEXT_FRESH_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": module.RUN_ID,
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_command_included": False,
            "files": {
                name: {
                    "bytes": (root / name).stat().st_size,
                    "sha256": module.sha256_file(root / name),
                }
                for name in names
            },
        },
    )
    return root


def test_deployment_runtime_verifies_and_fails_closed_on_tamper(tmp_path: Path) -> None:
    module = load_module()
    root = staged_runtime(tmp_path / "runtime", module)
    report = module.verify_dataset(root)
    assert report["member_count"] == 2
    assert report["authorized_for_full_candidate_evaluation"] is True
    assert report["authorized_for_submission"] is False
    (root / "graph_context_model_00.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="policy is ineligible|file changed"):
        module.verify_dataset(root)
