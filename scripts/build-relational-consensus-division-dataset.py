#!/usr/bin/env python
"""Stage a hash-bound runtime for a development-positive relational policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.temporal_contrastive.score_relational_division_development_probe import (
    EXPECTED_PARAMETER_COUNT,
    RUN_ID as PROBE_RUN_ID,
    validate_policy,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "indarkarhana/biohub-relational-consensus-division-v1"
RUN_ID = "competition-relational-consensus-division-dataset-v1"
POLICY_RUN_ID = "competition-relational-consensus-division-policy-v1"
DEVELOPMENT_RUN_ID = "competition-relational-division-development-v1"
MORPHOLOGY_RUN_ID = "competition-real-handcrafted-division-gate-v1"
SKLEARN_VERSION = "1.9.0"
SKLEARN_WHEEL_NAME = (
    "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64."
    "manylinux_2_28_x86_64.whl"
)
RUNTIME_FILES = {
    "learned_division_recovery.py": ROOT / "research/learned_division_recovery.py",
    "relational_division_inference.py": ROOT
    / "research/temporal_contrastive/relational_division_inference.py",
    "relational_division_model.py": ROOT
    / "research/temporal_contrastive/relational_division_model.py",
    "multiscale_contextual_pair_fusion.py": ROOT
    / "research/temporal_contrastive/multiscale_contextual_pair_fusion.py",
    "contextual_pair_fusion.py": ROOT
    / "research/temporal_contrastive/contextual_pair_fusion.py",
    "pair_fusion.py": ROOT / "research/temporal_contrastive/pair_fusion.py",
    "patch_model.py": ROOT / "research/temporal_contrastive/patch_model.py",
    "transition_context.py": ROOT
    / "research/temporal_contrastive/transition_context.py",
    "handcrafted_division.py": ROOT / "research/train_handcrafted_division_gate.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_sources(
    *,
    results_root: Path,
    probe_path: Path,
    morphology_root: Path,
    development_path: Path,
) -> dict[str, Any]:
    terminal, members = validate_policy(results_root)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    morphology_terminal_path = morphology_root / "handcrafted_division_gate_terminal.json"
    morphology_model_path = morphology_root / "handcrafted_division_gate.joblib"
    morphology = json.loads(morphology_terminal_path.read_text(encoding="utf-8"))
    if not (
        probe.get("schema_version") == 1
        and probe.get("status") == "development_probe_complete"
        and probe.get("run_id") == PROBE_RUN_ID
        and probe.get("selection_policy") == terminal["deployment_policy"]
        and probe.get("member_count") == len(members)
        and [row["model_sha256"] for row in probe.get("members", [])]
        == [row["model_sha256"] for row in members]
        and probe.get("absolute_threshold_used") is False
        and probe.get("weights_searched_on_probe") is False
        and probe.get("model_subset_searched_on_probe") is False
        and probe.get("competition_test_data_read") is False
        and probe.get("public_leaderboard_used_for_selection") is False
        and probe.get("submission_created") is False
        and probe.get("authorized_for_submission") is False
        and development.get("schema_version") == 1
        and development.get("status") == "development_positive"
        and development.get("run_id") == DEVELOPMENT_RUN_ID
        and development.get("relational_probe_sha256") == sha256_file(probe_path)
        and development.get("selected") == 3
        and development.get("tp") == 3
        and development.get("fp") == 0
        and development.get("absolute_threshold_used") is False
        and development.get("competition_test_data_read") is False
        and development.get("public_leaderboard_used_for_selection") is False
        and development.get("submission_created") is False
        and development.get("authorized_for_full_candidate_evaluation") is True
        and development.get("authorized_for_submission") is False
        and morphology.get("schema_version") == 1
        and morphology.get("status") == "completed"
        and morphology.get("run_id") == MORPHOLOGY_RUN_ID
        and morphology.get("feature_count") == 132
        and morphology.get("model_sha256") == sha256_file(morphology_model_path)
        and morphology.get("competition_test_data_read") is False
        and morphology.get("public_code_copied") is False
        and morphology.get("public_predictions_copied") is False
        and morphology.get("public_leaderboard_used_for_selection") is False
        and morphology.get("submission_created") is False
    ):
        raise ValueError("relational consensus source evidence is ineligible")
    return {
        "terminal": terminal,
        "members": members,
        "morphology_terminal_path": morphology_terminal_path,
        "morphology_model_path": morphology_model_path,
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "RELATIONAL_CONSENSUS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads((root / "relational-consensus-policy.json").read_text(encoding="utf-8"))
    member_names = {str(row["path"]) for row in policy.get("relational_members", [])}
    expected = {
        *RUNTIME_FILES,
        *member_names,
        "selection_audit_terminal.json",
        "relational_development_probe.json",
        "relational_development_evidence.json",
        "morphology_division_model.joblib",
        "morphology_training_terminal.json",
        "relational-consensus-policy.json",
        SKLEARN_WHEEL_NAME,
    }
    files = manifest.get("files", {})
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUN_ID
        and set(files) == expected
        and 1 <= len(member_names) <= 8
        and len(member_names) == len(policy.get("relational_members", []))
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
    ):
        raise ValueError("relational consensus manifest is invalid")
    for name, record in files.items():
        if sha256_file(root / name) != record.get("sha256"):
            raise ValueError(f"relational consensus runtime changed: {name}")
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "development_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("relational_policy")
        in {"equal_rank_selection_admitted_ensemble", "strongest_selection_individual"}
        and policy.get("relational_member_count") == len(member_names)
        and all(
            row.get("model_sha256") == sha256_file(root / row["path"])
            and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("selection_gate_passed") is True
            and row.get("audit_gate_passed") is True
            for row in policy.get("relational_members", [])
        )
        and policy.get("morphology_model_sha256") == sha256_file(root / "morphology_division_model.joblib")
        and policy.get("sklearn_version") == SKLEARN_VERSION
        and policy.get("sklearn_wheel_sha256") == sha256_file(root / SKLEARN_WHEEL_NAME)
        and policy.get("biological_geometry_minimum") == 3.0
        and policy.get("maximum_added_edges_per_movie") == 1
        and policy.get("base_safe_division_heuristic_enabled") is True
        and policy.get("external_policy_additive_only") is True
        and policy.get("exact_two_t4_required") is True
        and policy.get("absolute_threshold_used") is False
        and policy.get("model_subset_searched_on_audit") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
    ):
        raise ValueError("relational consensus policy changed")
    return {
        "status": "verified",
        "files": len(files),
        "relational_member_count": len(member_names),
        "manifest_sha256": sha256_file(manifest_path),
        "authorized_for_candidate_attachment": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--morphology-root", type=Path, required=True)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--sklearn-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.sklearn_wheel.name != SKLEARN_WHEEL_NAME:
        raise ValueError("relational runtime requires the pinned CPython 3.12 wheel")
    evidence = validate_sources(
        results_root=args.results_root,
        probe_path=args.probe,
        morphology_root=args.morphology_root,
        development_path=args.development_evidence,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    copies: dict[str, Path] = {
        **RUNTIME_FILES,
        "selection_audit_terminal.json": args.results_root / "relational_division_sweep_terminal.json",
        "relational_development_probe.json": args.probe,
        "relational_development_evidence.json": args.development_evidence,
        "morphology_division_model.joblib": evidence["morphology_model_path"],
        "morphology_training_terminal.json": evidence["morphology_terminal_path"],
        SKLEARN_WHEEL_NAME: args.sklearn_wheel,
    }
    member_records = []
    for index, member in enumerate(evidence["members"]):
        name = f"relational_division_model_{index:02d}.pt"
        copies[name] = member["checkpoint"]
        member_records.append(
            {
                "path": name,
                "member": member["member"],
                "seed": int(member["seed"]),
                "model_sha256": member["model_sha256"],
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "selection_average_precision": member["selection_average_precision"],
                "audit_average_precision": member["audit_average_precision"],
                "selection_gate_passed": True,
                "audit_gate_passed": True,
            }
        )
    for name, source in copies.items():
        shutil.copy2(source, args.output_root / name)
    policy_path = args.output_root / "relational-consensus-policy.json"
    write_json(
        policy_path,
        {
            "schema_version": 1,
            "status": "development_accepted",
            "run_id": POLICY_RUN_ID,
            "relational_policy": evidence["terminal"]["deployment_policy"],
            "relational_member_count": len(member_records),
            "relational_members": member_records,
            "morphology_model_sha256": sha256_file(evidence["morphology_model_path"]),
            "sklearn_version": SKLEARN_VERSION,
            "sklearn_wheel_sha256": sha256_file(args.sklearn_wheel),
            "biological_geometry_minimum": 3.0,
            "maximum_added_edges_per_movie": 1,
            "selection_rule": "identical top geometry-eligible candidate under precommitted relational equal-rank and independent temporal morphology rankings",
            "base_safe_division_heuristic_enabled": True,
            "external_policy_additive_only": True,
            "exact_two_t4_required": True,
            "absolute_threshold_used": False,
            "model_subset_searched_on_audit": False,
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
    names = [*copies, policy_path.name]
    manifest_path = args.output_root / "RELATIONAL_CONSENSUS_MANIFEST.json"
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
                name: {"path": name, "sha256": sha256_file(args.output_root / name)}
                for name in names
            },
        },
    )
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub Relational Consensus Division v1",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_dataset(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
