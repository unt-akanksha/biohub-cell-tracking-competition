#!/usr/bin/env python
"""Stage the development-positive ranked consensus division runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "indarkarhana/biohub-ranked-consensus-division-v1"
RUN_ID = "competition-ranked-consensus-division-dataset-v1"
POLICY_RUN_ID = "competition-ranked-consensus-division-policy-v1"
DEEP_RUN_ID = "competition-real-division-gate-v1"
MORPHOLOGY_RUN_ID = "competition-real-handcrafted-division-gate-v1"
DEVELOPMENT_RUN_ID = "competition-ranked-consensus-division-development-v1"
SKLEARN_VERSION = "1.9.0"
SKLEARN_WHEEL_NAME = (
    "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64."
    "manylinux_2_28_x86_64.whl"
)
RUNTIME_FILES = {
    "learned_division_recovery.py": ROOT / "research/learned_division_recovery.py",
    "multiscale_contextual_pair_fusion.py": ROOT
    / "research/temporal_contrastive/multiscale_contextual_pair_fusion.py",
    "patch_model.py": ROOT / "research/temporal_contrastive/patch_model.py",
    "handcrafted_division.py": ROOT / "research/train_handcrafted_division_gate.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_sources(
    deep_root: Path, morphology_root: Path, development_path: Path
) -> dict[str, Any]:
    deep_terminal_path = deep_root / "real_division_gate_terminal.json"
    deep_model_path = deep_root / "target_6bba/division_model.pt"
    morphology_terminal_path = morphology_root / "handcrafted_division_gate_terminal.json"
    morphology_model_path = morphology_root / "handcrafted_division_gate.joblib"
    deep = json.loads(deep_terminal_path.read_text(encoding="utf-8"))
    morphology = json.loads(morphology_terminal_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    deep_hash = sha256_file(deep_model_path)
    morphology_hash = sha256_file(morphology_model_path)
    weights = deep.get("ensemble_weights", {})
    pooled = development.get("pooled", {})
    if not (
        deep.get("schema_version") == 1
        and deep.get("status") == "accepted_at_selection"
        and deep.get("run_id") == DEEP_RUN_ID
        and deep.get("selection_gate_passed") is True
        and deep.get("final_probe_opened") is False
        and deep.get("competition_test_data_read") is False
        and deep.get("public_code_copied") is False
        and deep.get("public_predictions_copied") is False
        and deep.get("public_leaderboard_used_for_selection") is False
        and weights == {"target_44b6": 0.0, "target_6bba": 1.0}
        and deep.get("folds", {}).get("target_6bba", {}).get("model_sha256")
        == deep_hash
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
        and development.get("selected") == 3
        and development.get("tp") == 3
        and development.get("fp") == 0
        and development.get("absolute_threshold_used") is False
        and development.get("competition_test_data_read") is False
        and development.get("public_leaderboard_used_for_selection") is False
        and development.get("authorized_for_full_candidate_evaluation") is True
        and development.get("authorized_for_submission") is False
        and float(pooled.get("edge_after", {}).get("jaccard", math.nan))
        > float(pooled.get("edge_before", {}).get("jaccard", math.nan))
        and int(pooled.get("division_after", {}).get("tp", -1))
        > int(pooled.get("division_before", {}).get("tp", -1))
    ):
        raise ValueError("ranked consensus source evidence is ineligible")
    return {
        "deep_terminal_path": deep_terminal_path,
        "deep_model_path": deep_model_path,
        "deep_model_sha256": deep_hash,
        "morphology_terminal_path": morphology_terminal_path,
        "morphology_model_path": morphology_model_path,
        "morphology_model_sha256": morphology_hash,
        "development": development,
    }


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "RANKED_CONSENSUS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    expected = {
        *RUNTIME_FILES,
        "deep_division_model.pt",
        "deep_training_terminal.json",
        "morphology_division_model.joblib",
        "morphology_training_terminal.json",
        "development_evidence.json",
        "ranked-consensus-policy.json",
        SKLEARN_WHEEL_NAME,
    }
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUN_ID
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and set(files) == expected
    ):
        raise ValueError("ranked consensus manifest is invalid")
    for name, record in files.items():
        if sha256_file(root / name) != record.get("sha256"):
            raise ValueError(f"ranked consensus staged file changed: {name}")
    policy = json.loads((root / "ranked-consensus-policy.json").read_text())
    if not (
        policy.get("status") == "development_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("absolute_threshold_used") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
        and policy.get("deep_model_sha256")
        == sha256_file(root / "deep_division_model.pt")
        and policy.get("morphology_model_sha256")
        == sha256_file(root / "morphology_division_model.joblib")
        and policy.get("sklearn_version") == SKLEARN_VERSION
        and policy.get("sklearn_wheel_sha256")
        == sha256_file(root / SKLEARN_WHEEL_NAME)
    ):
        raise ValueError("ranked consensus policy changed")
    return {
        "status": "verified",
        "files": len(files),
        "manifest_sha256": sha256_file(manifest_path),
        "authorized_for_candidate_attachment": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-root", type=Path, required=True)
    parser.add_argument("--morphology-root", type=Path, required=True)
    parser.add_argument("--development-evidence", type=Path, required=True)
    parser.add_argument("--sklearn-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.sklearn_wheel.name != SKLEARN_WHEEL_NAME:
        raise ValueError("ranked consensus requires the pinned CPython 3.12 wheel")
    evidence = validate_sources(
        args.deep_root, args.morphology_root, args.development_evidence
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    copies = {
        **RUNTIME_FILES,
        "deep_division_model.pt": evidence["deep_model_path"],
        "deep_training_terminal.json": evidence["deep_terminal_path"],
        "morphology_division_model.joblib": evidence["morphology_model_path"],
        "morphology_training_terminal.json": evidence["morphology_terminal_path"],
        "development_evidence.json": args.development_evidence,
        SKLEARN_WHEEL_NAME: args.sklearn_wheel,
    }
    for name, source in copies.items():
        shutil.copy2(source, args.output_root / name)
    write_json(
        args.output_root / "ranked-consensus-policy.json",
        {
            "schema_version": 1,
            "status": "development_accepted",
            "run_id": POLICY_RUN_ID,
            "deep_model_sha256": evidence["deep_model_sha256"],
            "morphology_model_sha256": evidence["morphology_model_sha256"],
            "sklearn_version": SKLEARN_VERSION,
            "sklearn_wheel_sha256": sha256_file(args.sklearn_wheel),
            "biological_geometry_minimum": 3.0,
            "maximum_added_edges_per_movie": 1,
            "selection_rule": (
                "top geometry-eligible parent must agree under independent deep "
                "and temporal morphology rankings"
            ),
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
    names = [*copies, "ranked-consensus-policy.json"]
    write_json(
        args.output_root / "RANKED_CONSENSUS_MANIFEST.json",
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
            "title": "Biohub Ranked Consensus Division v1",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_dataset(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
