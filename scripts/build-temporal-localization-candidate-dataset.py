#!/usr/bin/env python
"""Stage the synthetic-gated temporal localization runtime as a private dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "indarkarhana/biohub-temporal-localization-consensus-v1"
RUN_ID = "competition-temporal-localization-consensus-dataset-v1"
POLICY_RUN_ID = "competition-temporal-localization-consensus-policy-v1"
TRAINING_RUN_ID = "synthetic16-temporal-node-localizer-v1"
DEVELOPMENT_RUN_ID = "temporal-node-localizer-real-development-v1"
EXPECTED_PARAMETER_COUNT = 71_249_805
MANIFEST_NAME = "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json"
POLICY_NAME = "temporal-localization-consensus-policy.json"
RUNTIME_FILES = {
    "research/temporal_contrastive/patch_model.py": ROOT
    / "research/temporal_contrastive/patch_model.py",
    "research/temporal_localization/__init__.py": ROOT
    / "research/temporal_localization/__init__.py",
    "research/temporal_localization/model.py": ROOT
    / "research/temporal_localization/model.py",
    "research/temporal_localization/inference.py": ROOT
    / "research/temporal_localization/inference.py",
    "research/temporal_localization/consensus.py": ROOT
    / "research/temporal_localization/consensus.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def accepted_members(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for terminal_path in sorted(results_root.glob("gpu_*/member_*/worker_terminal.json")):
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        checkpoint = terminal_path.parent / "localization_model.pt"
        if terminal.get("status") != "completed":
            continue
        if not (
            terminal.get("schema_version") == 1
            and terminal.get("run_id") == TRAINING_RUN_ID
            and terminal.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and terminal.get("selection_gate_passed") is True
            and terminal.get("audit_gate_passed") is True
            and terminal.get("checkpoint_frozen_before_audit") is True
            and terminal.get("audit_opened") is True
            and checkpoint.is_file()
            and terminal.get("model_sha256") == sha256_file(checkpoint)
            and terminal.get("competition_data_read") is False
            and terminal.get("public_code_copied") is False
            and terminal.get("public_predictions_copied") is False
            and terminal.get("public_leaderboard_used_for_selection") is False
            and terminal.get("submission_created") is False
        ):
            raise ValueError(f"accepted localization member evidence changed: {terminal_path}")
        records.append(
            {
                "terminal_path": terminal_path,
                "checkpoint": checkpoint,
                "seed": int(terminal["seed"]),
                "model_sha256": terminal["model_sha256"],
                "selection": terminal["best_selection"],
                "audit": terminal["final_audit"],
            }
        )
    if not 3 <= len(records) <= 4:
        raise ValueError(f"localization deployment requires 3 or 4 accepted members, saw {len(records)}")
    if len({row["seed"] for row in records}) != len(records) or len(
        {row["model_sha256"] for row in records}
    ) != len(records):
        raise ValueError("accepted localization members are not independent")
    return records


def validate_sources(results_root: Path) -> dict[str, Any]:
    members = accepted_members(results_root)
    development_path = results_root / "real-development-probe.json"
    development = json.loads(development_path.read_text(encoding="utf-8"))
    expected_members = [
        {"seed": row["seed"], "model_sha256": row["model_sha256"]} for row in members
    ]
    if not (
        development.get("schema_version") == 1
        and development.get("status") == "development_passed"
        and development.get("run_id") == DEVELOPMENT_RUN_ID
        and development.get("training_run_id") == TRAINING_RUN_ID
        and development.get("members") == expected_members
        and development.get("gate", {}).get("passed") is True
        and development.get("gate", {}).get("no_movie_recall_regression") is True
        and int(development.get("gate", {}).get("matched_node_gain", 0)) > 0
        and development.get("gate", {}).get("mean_movie_matched_distance_improved") is True
        and development.get("gate", {}).get("all_global_move_fraction_gates_passed") is True
        and development.get("acceptance_labels_already_opened") is True
        and development.get("development_only") is True
        and development.get("policy_or_member_selection_performed") is False
        and development.get("competition_test_data_read") is False
        and development.get("public_leaderboard_used_for_selection") is False
        and development.get("submission_created") is False
        and development.get("authorized_for_submission") is False
    ):
        raise ValueError("real temporal-localization development evidence is ineligible")
    return {"members": members, "development": development, "development_path": development_path}


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    policy_path = root / POLICY_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    members = policy.get("localization_members", [])
    expected = {
        *RUNTIME_FILES,
        POLICY_NAME,
        "evidence/real-development-probe.json",
        *[str(row["path"]) for row in members],
        *[str(row["terminal_path"]) for row in members],
    }
    files = manifest.get("files", {})
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUN_ID
        and set(files) == expected
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
    ):
        raise ValueError("temporal localization runtime manifest is invalid")
    for name, record in files.items():
        path = root / name
        if not (
            path.is_file()
            and record.get("path") == name
            and record.get("sha256") == sha256_file(path)
        ):
            raise ValueError(f"temporal localization runtime changed: {name}")
    hashes = [row.get("model_sha256") for row in members]
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "development_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and 3 <= len(members) <= 4
        and policy.get("localization_member_count") == len(members)
        and len(set(hashes)) == len(members)
        and all(
            row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("selection_gate_passed") is True
            and row.get("audit_gate_passed") is True
            and row.get("model_sha256") == sha256_file(root / row["path"])
            for row in members
        )
        and policy.get("ensemble_policy") == "equal_mean_all_synthetic_eligible_members"
        and policy.get("minimum_members") == 3
        and policy.get("blend") == 0.75
        and policy.get("minimum_correction_um") == 2.0
        and policy.get("maximum_correction_um") == 9.5
        and policy.get("maximum_member_disagreement_um") == 1.5
        and policy.get("maximum_predicted_sigma_um") == 2.5
        and policy.get("maximum_safe_probability") == 0.35
        and policy.get("minimum_direction_cosine") == 0.8
        and policy.get("maximum_move_fraction") == 0.1
        and policy.get("node_count_preserving") is True
        and policy.get("topology_preserving") is True
        and policy.get("exact_two_t4_required") is True
        and policy.get("model_subset_searched_on_audit") is False
        and policy.get("weights_searched_on_development") is False
        and policy.get("threshold_searched_on_development") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
    ):
        raise ValueError("temporal localization runtime policy is invalid")
    return {
        "status": "verified",
        "files": len(files),
        "localization_member_count": len(members),
        "manifest_sha256": sha256_file(manifest_path),
        "authorized_for_candidate_attachment": True,
        "authorized_for_submission": False,
    }


def stage_dataset(results_root: Path, output_root: Path) -> dict[str, Any]:
    evidence = validate_sources(results_root)
    output_root.mkdir(parents=True, exist_ok=False)
    copies: dict[str, Path] = {
        **RUNTIME_FILES,
        "evidence/real-development-probe.json": evidence["development_path"],
    }
    member_records: list[dict[str, Any]] = []
    for index, member in enumerate(evidence["members"]):
        model_name = f"models/temporal_localizer_{index:02d}.pt"
        terminal_name = f"evidence/member_{index:02d}_terminal.json"
        copies[model_name] = member["checkpoint"]
        copies[terminal_name] = member["terminal_path"]
        member_records.append(
            {
                "path": model_name,
                "terminal_path": terminal_name,
                "seed": member["seed"],
                "model_sha256": member["model_sha256"],
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "selection_mean_residual_um": member["selection"]["mean_residual_um"],
                "audit_mean_residual_um": member["audit"]["mean_residual_um"],
                "selection_gate_passed": True,
                "audit_gate_passed": True,
            }
        )
    for name, source in copies.items():
        destination = output_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    policy_path = output_root / POLICY_NAME
    write_json(
        policy_path,
        {
            "schema_version": 1,
            "status": "development_accepted",
            "run_id": POLICY_RUN_ID,
            "localization_member_count": len(member_records),
            "localization_members": member_records,
            "ensemble_policy": "equal_mean_all_synthetic_eligible_members",
            "minimum_members": 3,
            "blend": 0.75,
            "minimum_correction_um": 2.0,
            "maximum_correction_um": 9.5,
            "maximum_member_disagreement_um": 1.5,
            "maximum_predicted_sigma_um": 2.5,
            "maximum_safe_probability": 0.35,
            "minimum_direction_cosine": 0.8,
            "maximum_move_fraction": 0.1,
            "node_count_preserving": True,
            "topology_preserving": True,
            "exact_two_t4_required": True,
            "model_subset_searched_on_audit": False,
            "weights_searched_on_development": False,
            "threshold_searched_on_development": False,
            "competition_train_data_read": True,
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_full_candidate_evaluation": True,
            "authorized_for_submission": False,
        },
    )
    names = [*copies, POLICY_NAME]
    manifest_path = output_root / MANIFEST_NAME
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": RUN_ID,
            "competition_train_data_read": True,
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_command_included": False,
            "files": {
                name: {"path": name, "sha256": sha256_file(output_root / name)}
                for name in names
            },
        },
    )
    write_json(
        output_root / "dataset-metadata.json",
        {
            "title": "Biohub Temporal Localization Consensus v1",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    return verify_dataset(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_dataset(args.results_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
