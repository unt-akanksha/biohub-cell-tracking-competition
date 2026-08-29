#!/usr/bin/env python
"""Stage accepted Antelume focused-division weights as a private dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "focused-division-gate-dataset-v1"
TRAINING_RUN_ID = "focused-division-gate-v1"
POLICY_RUN_ID = "external-division-recovery-policy-v1"
DATASET_ID = "indarkarhana/biohub-focused-division-gate-v1"
FOLDS = ("target_44b6", "target_6bba")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source(root: Path) -> dict[str, Any]:
    terminal_path = root / "focused_division_gate_terminal.json"
    policy_path = root / "division-recovery-policy.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    model_hashes = policy.get("model_sha256", {})
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "accepted"
        and terminal.get("run_id") == TRAINING_RUN_ID
        and terminal.get("execution_gpu_count") == 1
        and terminal.get("both_folds_selected") is True
        and terminal.get("audit_opened") is True
        and terminal.get("audit_opened_after_threshold_freeze") is True
        and terminal.get("authorized_for_competition_graph_evaluation") is True
        and terminal.get("authorized_for_submission") is False
        and terminal.get("competition_data_read") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and policy.get("schema_version") == 1
        and policy.get("status") == "accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("model_training_run_id") == TRAINING_RUN_ID
        and policy.get("audit_opened_after_threshold_freeze") is True
        and policy.get("authorized_for_competition_graph_evaluation") is True
        and policy.get("authorized_for_submission") is False
        and policy.get("competition_data_read") is False
        and policy.get("public_leaderboard_used_for_selection") is False
        and policy.get("submission_created") is False
        and terminal.get("model_sha256") == model_hashes
        and set(model_hashes) == set(FOLDS)
    ):
        raise ValueError("focused division source evidence is ineligible")
    for fold in FOLDS:
        worker = terminal.get("folds", {}).get(fold, {})
        checkpoint = root / fold / "division_model.pt"
        if not (
            worker.get("status") == "accepted_at_selection"
            and worker.get("selection_gate_passed") is True
            and worker.get("checkpoint_frozen_before_audit") is True
            and worker.get("model_sha256") == model_hashes[fold]
            and checkpoint.is_file()
            and sha256_file(checkpoint) == model_hashes[fold]
        ):
            raise ValueError(f"focused division fold is ineligible: {fold}")
    return {"terminal": terminal, "policy": policy}


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "FOCUSED_DIVISION_GATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    expected = {
        "focused_division_gate_terminal.json",
        "division-recovery-policy.json",
        *(f"{fold}/division_model.pt" for fold in FOLDS),
    }
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("run_id") == RUN_ID
        and manifest.get("training_run_id") == TRAINING_RUN_ID
        and manifest.get("policy_run_id") == POLICY_RUN_ID
        and manifest.get("competition_data_read") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and set(files) == expected
    ):
        raise ValueError("focused division dataset manifest is invalid")
    for name, record in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"focused division staged file changed: {name}")
    validate_source(root)
    return {
        "run_id": RUN_ID,
        "files": len(files),
        "manifest_sha256": sha256_file(manifest_path),
        "authorized_for_candidate_attachment": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    validate_source(args.source_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    for name in (
        "focused_division_gate_terminal.json",
        "division-recovery-policy.json",
    ):
        shutil.copy2(args.source_root / name, args.output_root / name)
    for fold in FOLDS:
        target = args.output_root / fold
        target.mkdir()
        shutil.copy2(
            args.source_root / fold / "division_model.pt",
            target / "division_model.pt",
        )
    names = [
        "focused_division_gate_terminal.json",
        "division-recovery-policy.json",
        *(f"{fold}/division_model.pt" for fold in FOLDS),
    ]
    write_json(
        args.output_root / "FOCUSED_DIVISION_GATE_MANIFEST.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "training_run_id": TRAINING_RUN_ID,
            "policy_run_id": POLICY_RUN_ID,
            "competition_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_command_included": False,
            "files": {
                name: {"path": name, "sha256": sha256_file(args.output_root / name)}
                for name in names
            },
        },
    )
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub Focused Division Gate v1",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_dataset(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
