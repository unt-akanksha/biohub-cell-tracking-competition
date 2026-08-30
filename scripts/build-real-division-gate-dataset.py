#!/usr/bin/env python
"""Stage an accepted Antelume real-division gate as a private dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any


RUN_ID = "competition-real-division-gate-dataset-v1"
TRAINING_RUN_ID = "competition-real-division-gate-v1"
PROBE_RUN_ID = "competition-train-focused-division-transfer-probe-v1"
POLICY_RUN_ID = "competition-real-division-recovery-policy-v1"
DATASET_ID = "indarkarhana/biohub-real-division-gate-v1"
FOLDS = ("target_44b6", "target_6bba")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source(source_root: Path, probe_path: Path) -> dict[str, Any]:
    terminal_path = source_root / "real_division_gate_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    folds = terminal.get("folds", {})
    model_hashes = {fold: folds.get(fold, {}).get("model_sha256") for fold in FOLDS}
    threshold = terminal.get("frozen_division_logit_threshold")
    policy = probe.get("policy_evaluation", {})
    conjunctive = policy.get("conjunctive", {})
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "accepted_at_selection"
        and terminal.get("run_id") == TRAINING_RUN_ID
        and terminal.get("execution_gpu_count") == 1
        and terminal.get("selection_gate_passed") is True
        and terminal.get("final_probe_opened") is False
        and terminal.get("checkpoint_frozen_before_final_probe") is True
        and terminal.get("competition_train_data_read") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_final_probe") is True
        and isinstance(threshold, (int, float))
        and math.isfinite(float(threshold))
        and set(folds) == set(FOLDS)
        and len(set(model_hashes.values())) == 2
        and probe.get("schema_version") == 1
        and probe.get("status") == "diagnostic_complete"
        and probe.get("run_id") == PROBE_RUN_ID
        and probe.get("model_sha256") == [model_hashes[fold] for fold in FOLDS]
        and probe.get("competition_train_data_read") is True
        and probe.get("competition_test_data_read") is False
        and probe.get("public_leaderboard_used_for_selection") is False
        and probe.get("threshold_selected") is False
        and probe.get("final_probe_opened") is True
        and probe.get("final_probe_opened_after_training_threshold_freeze") is True
        and probe.get("submission_created") is False
        and probe.get("authorized_for_competition_graph_evaluation") is True
        and probe.get("authorized_for_submission") is False
        and policy.get("status") == "accepted"
        and policy.get("training_terminal_sha256") == sha256_file(terminal_path)
        and float(policy.get("model_threshold_frozen_before_probe", math.nan))
        == float(threshold)
        and float(policy.get("biological_geometry_minimum", math.nan)) == 3.0
        and policy.get("authorized_for_competition_graph_evaluation") is True
        and policy.get("authorized_for_submission") is False
        and int(conjunctive.get("tp", -1)) >= 2
        and int(conjunctive.get("fp", -1)) >= 0
        and float(conjunctive.get("precision", -1.0)) >= 0.75
        and float(conjunctive.get("jaccard", -1.0)) > 0.0
    ):
        raise ValueError("real division source evidence is ineligible")
    for fold in FOLDS:
        worker = folds[fold]
        checkpoint = source_root / fold / "division_model.pt"
        if not (
            worker.get("status") == "completed"
            and worker.get("checkpoint_frozen_before_final_probe") is True
            and worker.get("competition_test_data_read") is False
            and worker.get("model_sha256") == model_hashes[fold]
            and checkpoint.is_file()
            and sha256_file(checkpoint) == model_hashes[fold]
        ):
            raise ValueError(f"real division fold is ineligible: {fold}")
    return {
        "terminal": terminal,
        "probe": probe,
        "model_hashes": model_hashes,
        "threshold": float(threshold),
        "geometry_minimum": 3.0,
    }


def build_policy(evidence: dict[str, Any], probe_sha256: str) -> dict[str, Any]:
    terminal = evidence["terminal"]
    probe = evidence["probe"]
    return {
        "schema_version": 1,
        "status": "accepted",
        "run_id": POLICY_RUN_ID,
        "model_training_run_id": TRAINING_RUN_ID,
        "appearance_family": terminal["appearance_family"],
        "focused_division_family": terminal["family"],
        "model_sha256": evidence["model_hashes"],
        "ensemble": "mean of two independently initialized real-domain division gates",
        "frozen_division_logit_threshold": evidence["threshold"],
        "biological_geometry_minimum": evidence["geometry_minimum"],
        "selection": terminal["threshold_selection"],
        "held_out_probe": probe["policy_evaluation"]["conjunctive"],
        "probe_result_sha256": probe_sha256,
        "final_probe_opened_after_threshold_freeze": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_competition_graph_evaluation": True,
        "authorized_for_submission": False,
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "REAL_DIVISION_GATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    expected = {
        "real_division_gate_terminal.json",
        "competition_division_probe_result.json",
        "division-recovery-policy.json",
        *(f"{fold}/division_model.pt" for fold in FOLDS),
    }
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("run_id") == RUN_ID
        and manifest.get("training_run_id") == TRAINING_RUN_ID
        and manifest.get("policy_run_id") == POLICY_RUN_ID
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and set(files) == expected
    ):
        raise ValueError("real division dataset manifest is invalid")
    for name, record in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"real division staged file changed: {name}")
    policy = json.loads((root / "division-recovery-policy.json").read_text())
    if not (
        policy.get("status") == "accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("authorized_for_competition_graph_evaluation") is True
        and policy.get("authorized_for_submission") is False
        and policy.get("model_sha256")
        == {
            fold: sha256_file(root / fold / "division_model.pt") for fold in FOLDS
        }
        and policy.get("probe_result_sha256")
        == sha256_file(root / "competition_division_probe_result.json")
    ):
        raise ValueError("real division staged policy changed")
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
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    evidence = validate_source(args.source_root, args.probe_result)
    args.output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(
        args.source_root / "real_division_gate_terminal.json",
        args.output_root / "real_division_gate_terminal.json",
    )
    shutil.copy2(
        args.probe_result,
        args.output_root / "competition_division_probe_result.json",
    )
    for fold in FOLDS:
        target = args.output_root / fold
        target.mkdir()
        shutil.copy2(
            args.source_root / fold / "division_model.pt",
            target / "division_model.pt",
        )
    policy = build_policy(evidence, sha256_file(args.probe_result))
    write_json(args.output_root / "division-recovery-policy.json", policy)
    names = [
        "real_division_gate_terminal.json",
        "competition_division_probe_result.json",
        "division-recovery-policy.json",
        *(f"{fold}/division_model.pt" for fold in FOLDS),
    ]
    write_json(
        args.output_root / "REAL_DIVISION_GATE_MANIFEST.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "training_run_id": TRAINING_RUN_ID,
            "policy_run_id": POLICY_RUN_ID,
            "competition_train_data_read": True,
            "competition_test_data_read": False,
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
            "title": "Biohub Real Division Gate v1",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_dataset(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
