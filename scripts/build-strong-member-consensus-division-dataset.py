#!/usr/bin/env python
"""Stage a hash-bound runtime for an admitted overnight deep policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.temporal_contrastive.overnight_seed_policy import (
    resolve_member_paths,
    select_precommitted_members,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "indarkarhana/biohub-strong-member-consensus-division-v2"
RUN_ID = "competition-strong-member-consensus-division-dataset-v2"
POLICY_RUN_ID = "competition-strong-member-consensus-division-policy-v2"
PROBE_RUN_ID = "competition-real-division-seed-ensemble-probe-v1"
DEVELOPMENT_RUN_ID = "competition-ranked-consensus-division-development-v1"
MORPHOLOGY_RUN_ID = "competition-real-handcrafted-division-gate-v1"
SKLEARN_VERSION = "1.9.0"
SKLEARN_WHEEL_NAME = (
    "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64."
    "manylinux_2_28_x86_64.whl"
)
RUNTIME_FILES = {
    "learned_division_recovery.py": ROOT / "research/learned_division_recovery.py",
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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def member_record(member: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "path": name,
        "seed": int(member["seed"]),
        "fold": str(member["fold"]),
        "model_sha256": str(member["model_sha256"]),
        "selection_average_precision": float(
            member["selection"]["average_precision"]
        ),
        "parameter_count": int(member["parameter_count"]),
        "trainable_parameters": int(member["trainable_parameters"]),
    }


def validate_sources(
    *,
    sweep_root: Path,
    ensemble_terminal_path: Path,
    probe_path: Path,
    morphology_root: Path,
    development_path: Path,
) -> dict[str, Any]:
    ensemble_terminal = json.loads(ensemble_terminal_path.read_text(encoding="utf-8"))
    policy_name, members = select_precommitted_members(ensemble_terminal)
    model_paths = resolve_member_paths(members, sweep_root)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    morphology_terminal_path = morphology_root / "handcrafted_division_gate_terminal.json"
    morphology_model_path = morphology_root / "handcrafted_division_gate.joblib"
    morphology = json.loads(morphology_terminal_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    morphology_hash = sha256_file(morphology_model_path)
    expected_probe_members = [
        {
            "seed": int(member["seed"]),
            "fold": str(member["fold"]),
            "model_sha256": str(member["model_sha256"]),
            "selection_average_precision": float(
                member["selection"]["average_precision"]
            ),
        }
        for member in members
    ]
    pooled = development.get("pooled", {})
    if not (
        1 <= len(members) <= 16
        and probe.get("schema_version") == 1
        and probe.get("status") == "development_probe_complete"
        and probe.get("run_id") == PROBE_RUN_ID
        and probe.get("selection_terminal_sha256")
        == sha256_file(ensemble_terminal_path)
        and probe.get("selection_policy") == policy_name
        and probe.get("member_count") == len(members)
        and probe.get("members") == expected_probe_members
        and probe.get("absolute_threshold_used") is False
        and probe.get("weights_searched_on_probe") is False
        and probe.get("model_subset_searched_on_probe") is False
        and probe.get("competition_test_data_read") is False
        and probe.get("public_leaderboard_used_for_selection") is False
        and probe.get("submission_created") is False
        and probe.get("authorized_for_ranked_consensus_development_evaluation")
        is True
        and probe.get("authorized_for_submission") is False
        and morphology.get("schema_version") == 1
        and morphology.get("status") == "completed"
        and morphology.get("run_id") == MORPHOLOGY_RUN_ID
        and morphology.get("feature_count") == 132
        and morphology.get("competition_test_data_read") is False
        and morphology.get("public_code_copied") is False
        and morphology.get("public_predictions_copied") is False
        and morphology.get("public_leaderboard_used_for_selection") is False
        and morphology.get("model_sha256") == morphology_hash
        and development.get("schema_version") == 1
        and development.get("status") == "development_positive"
        and development.get("run_id") == DEVELOPMENT_RUN_ID
        and development.get("deep_probe_sha256") == sha256_file(probe_path)
        and development.get("selected", 0) >= 1
        and development.get("tp") == development.get("selected")
        and development.get("fp") == 0
        and development.get("absolute_threshold_used") is False
        and development.get("competition_test_data_read") is False
        and development.get("public_leaderboard_used_for_selection") is False
        and development.get("submission_created") is False
        and development.get("authorized_for_full_candidate_evaluation") is True
        and development.get("authorized_for_submission") is False
        and float(pooled.get("edge_after", {}).get("jaccard", math.nan))
        > float(pooled.get("edge_before", {}).get("jaccard", math.nan))
        and int(pooled.get("division_after", {}).get("tp", -1))
        > int(pooled.get("division_before", {}).get("tp", -1))
        and int(pooled.get("division_after", {}).get("fp", -1))
        <= int(pooled.get("division_before", {}).get("fp", -1))
    ):
        raise ValueError("strong-member consensus source evidence is ineligible")
    return {
        "selection_policy": policy_name,
        "members": members,
        "model_paths": model_paths,
        "morphology_terminal_path": morphology_terminal_path,
        "morphology_model_path": morphology_model_path,
        "morphology_model_sha256": morphology_hash,
        "development": development,
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "STRONG_MEMBER_CONSENSUS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(
        (root / "strong-member-consensus-policy.json").read_text(encoding="utf-8")
    )
    deep_names = {str(row["path"]) for row in policy.get("deep_members", [])}
    expected = {
        *RUNTIME_FILES,
        *deep_names,
        "deep_selection_terminal.json",
        "deep_development_probe.json",
        "morphology_division_model.joblib",
        "morphology_training_terminal.json",
        "development_evidence.json",
        "strong-member-consensus-policy.json",
        SKLEARN_WHEEL_NAME,
    }
    files = manifest.get("files", {})
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUN_ID
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and set(files) == expected
        and 1 <= len(deep_names) <= 16
        and len(deep_names) == len(policy.get("deep_members", []))
    ):
        raise ValueError("strong-member consensus manifest is invalid")
    for name, record in files.items():
        if sha256_file(root / name) != record.get("sha256"):
            raise ValueError(f"strong-member runtime file changed: {name}")
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "development_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("deep_policy")
        in {"equal_rank_admitted_ensemble", "strongest_individual_rank"}
        and policy.get("deep_member_count") == len(deep_names)
        and all(
            row.get("model_sha256") == sha256_file(root / row["path"])
            and row.get("parameter_count") == 46_386_607
            and row.get("trainable_parameters") == 25_178_047
            for row in policy.get("deep_members", [])
        )
        and policy.get("base_safe_division_heuristic_enabled") is True
        and policy.get("external_policy_additive_only") is True
        and policy.get("maximum_added_edges_per_movie") == 1
        and policy.get("absolute_threshold_used") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
        and policy.get("morphology_model_sha256")
        == sha256_file(root / "morphology_division_model.joblib")
        and policy.get("sklearn_version") == SKLEARN_VERSION
        and policy.get("sklearn_wheel_sha256")
        == sha256_file(root / SKLEARN_WHEEL_NAME)
    ):
        raise ValueError("strong-member consensus policy changed")
    return {
        "status": "verified",
        "files": len(files),
        "deep_member_count": len(deep_names),
        "manifest_sha256": sha256_file(manifest_path),
        "authorized_for_candidate_attachment": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--ensemble-terminal", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--morphology-root", type=Path, required=True)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--sklearn-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.sklearn_wheel.name != SKLEARN_WHEEL_NAME:
        raise ValueError("strong-member runtime requires the pinned CPython 3.12 wheel")
    evidence = validate_sources(
        sweep_root=args.sweep_root,
        ensemble_terminal_path=args.ensemble_terminal,
        probe_path=args.probe_output,
        morphology_root=args.morphology_root,
        development_path=args.development_evidence,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    copies: dict[str, Path] = {
        **RUNTIME_FILES,
        "deep_selection_terminal.json": args.ensemble_terminal,
        "deep_development_probe.json": args.probe_output,
        "morphology_division_model.joblib": evidence["morphology_model_path"],
        "morphology_training_terminal.json": evidence["morphology_terminal_path"],
        "development_evidence.json": args.development_evidence,
        SKLEARN_WHEEL_NAME: args.sklearn_wheel,
    }
    deep_records = []
    for index, (member, source) in enumerate(
        zip(evidence["members"], evidence["model_paths"], strict=True)
    ):
        name = f"deep_division_model_{index:02d}.pt"
        copies[name] = source
        deep_records.append(member_record(member, name))
    for name, source in copies.items():
        shutil.copy2(source, args.output_root / name)
    policy_path = args.output_root / "strong-member-consensus-policy.json"
    write_json(
        policy_path,
        {
            "schema_version": 1,
            "status": "development_accepted",
            "run_id": POLICY_RUN_ID,
            "deep_policy": evidence["selection_policy"],
            "deep_member_count": len(deep_records),
            "deep_members": deep_records,
            "morphology_model_sha256": evidence["morphology_model_sha256"],
            "sklearn_version": SKLEARN_VERSION,
            "sklearn_wheel_sha256": sha256_file(args.sklearn_wheel),
            "biological_geometry_minimum": 3.0,
            "maximum_added_edges_per_movie": 1,
            "selection_rule": (
                "top geometry-eligible parent must agree under the frozen deep "
                "rank policy and independent temporal morphology ranking"
            ),
            "base_safe_division_heuristic_enabled": True,
            "external_policy_additive_only": True,
            "absolute_threshold_used": False,
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
    manifest_path = args.output_root / "STRONG_MEMBER_CONSENSUS_MANIFEST.json"
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
            "title": "Biohub Strong Member Consensus Division v2",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_dataset(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
