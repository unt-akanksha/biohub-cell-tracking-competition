#!/usr/bin/env python
"""Package a fixed equal-logit v1+v2 detector ensemble for clean validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "scripts" / "build-peak-rank-validation-runtime.py"))
SOURCES = BASE["SOURCES"]
TARGET_NAME = "biohub-peak-rank-logit-ensemble-validation-runtime-v4"
TARGET = ROOT / ".biohub" / "staging" / TARGET_NAME
DATASET_ID = f"indarkarhana/{TARGET_NAME}"
DATASET_TITLE = "Biohub Peak Rank Equal-Logit Ensemble Validation Runtime v4"
PURPOSE = "Two-GPU clean validation of a fixed equal-logit detector ensemble"
PARAMETER_COUNT = 76_762_956
RUN_ID = "clean-equal-logit-peak-rank-ensemble-v4"
FUSION = "equal_logit_and_offset_mean"
ARCHITECTURE_DESCRIPTION = (
    "equal-logit ensemble of independent temporal 3D ConvNeXt U-Net peak rankers"
)
MEMBERS = (
    {
        "name": "v1",
        "runtime": ROOT / ".biohub" / "staging" / "biohub-peak-rank-validation-runtime-v1",
        "controller": ROOT / ".biohub" / "automation" / "peak-rank-validation-controller-v1.json",
        "checkpoint_file": "member-v1.pt",
        "expected_parameter_count": 38_381_478,
    },
    {
        "name": "depth-pu-v2",
        "runtime": ROOT
        / ".biohub"
        / "staging"
        / "biohub-peak-rank-depth-pu-validation-runtime-v2",
        "controller": ROOT
        / ".biohub"
        / "automation"
        / "peak-rank-depth-pu-validation-controller-v2.json",
        "checkpoint_file": "member-depth-pu-v2.pt",
        "expected_parameter_count": 38_381_478,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checked_target() -> Path:
    staging = (ROOT / ".biohub" / "staging").resolve()
    target = TARGET.resolve()
    if target.parent != staging or target.name != TARGET_NAME:
        raise RuntimeError(f"unsafe ensemble staging target: {target}")
    return target


def unique_result(download_root: Path) -> Path:
    matches = sorted(download_root.rglob("peak_rank_validation.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one member validation result, saw {matches}")
    return matches[0]


def validate_member(spec: dict) -> dict:
    runtime = Path(spec["runtime"])
    controller_path = Path(spec["controller"])
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    manifest_path = runtime / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = runtime / "peak_rank_detector.pt"
    terminal_path = runtime / "training_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    result_path = unique_result(Path(controller["download_root"]))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_parameter_count = int(spec.get("expected_parameter_count", 38_381_478))
    model_family = terminal.get("model_family", "temporal_peak_rank_v1")
    architecture = manifest.get("architecture")
    architecture_is_project_detector = (
        isinstance(architecture, str)
        and "temporal 3D ConvNeXt U-Net" in architecture
        and "public" not in architecture.lower()
    )
    if not (
        controller.get("status") == "completed"
        and controller.get("accepted_for_candidate_integration") is True
        and controller.get("promotion_passed") is True
        and controller.get("competition_submission_performed") is False
        and controller.get("checkpoint_sha256") == manifest.get("checkpoint_sha256")
        and controller.get("validation_result_sha256") == sha256_file(result_path)
        and manifest.get("schema_version") == 1
        and architecture_is_project_detector
        and manifest.get("parameter_count") == expected_parameter_count
        and manifest.get("model_family", "temporal_peak_rank_v1") == model_family
        and manifest.get("ensemble_size") == 1
        and manifest.get("training_audit_passed") is True
        and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
        and manifest.get("widths") == terminal.get("widths")
        and manifest.get("depths") == terminal.get("depths")
        and terminal.get("status") == "accepted_at_audit"
        and terminal.get("audit_passed") is True
        and result.get("run_id") == "temporal-peak-rank-clean-validation-v1"
        and result.get("selection_passed") is True
        and result.get("acceptance_opened") is True
        and result.get("promotion_passed") is True
        and result.get("competition_test_data_read") is False
        and result.get("public_leaderboard_used_for_selection") is False
        and result.get("competition_submission_performed") is False
        and result.get("provenance", {}).get("checkpoint_sha256")
        == manifest.get("checkpoint_sha256")
    ):
        raise RuntimeError(f"ensemble member is not clean-promoted: {spec['name']}")
    for name, record in manifest.get("files", {}).items():
        path = runtime / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"ensemble member runtime changed: {spec['name']}:{name}")
    return {
        "name": spec["name"],
        "checkpoint_source": checkpoint,
        "checkpoint_file": spec["checkpoint_file"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "parameter_count": manifest["parameter_count"],
        "widths": manifest["widths"],
        "depths": manifest["depths"],
        "model_family": model_family,
        "training_terminal_sha256": sha256_file(terminal_path),
        "member_runtime_manifest_sha256": sha256_file(manifest_path),
        "clean_validation_sha256": sha256_file(result_path),
        "clean_validation_selected_tta_mode": result["selected_tta_mode"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    target = checked_target()
    members = [validate_member(spec) for spec in MEMBERS]
    equal_weight = 1.0 / len(members)
    for row in members:
        row["ensemble_weight"] = equal_weight
    if abs(sum(row["ensemble_weight"] for row in members) - 1.0) > 1e-12:
        raise RuntimeError("ensemble weights do not sum to one")
    if sum(row["parameter_count"] for row in members) != PARAMETER_COUNT:
        raise RuntimeError("ensemble member parameter total changed")
    if len({row["checkpoint_sha256"] for row in members}) != len(members):
        raise RuntimeError("ensemble checkpoints are not independent")
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} already exists; pass --replace")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name, source in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)
    terminal_members = []
    for row in members:
        destination = target / row["checkpoint_file"]
        shutil.copy2(row["checkpoint_source"], destination)
        if sha256_file(destination) != row["checkpoint_sha256"]:
            raise RuntimeError(f"copied ensemble member changed: {row['name']}")
        terminal_members.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"checkpoint_source"}
            }
        )
    checkpoint_manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "fusion": FUSION,
        "members": [
            {
                "name": row["name"],
                "checkpoint_file": row["checkpoint_file"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "ensemble_weight": row["ensemble_weight"],
            }
            for row in terminal_members
        ],
    }
    checkpoint_path = target / "peak_rank_detector.pt"
    write_json(checkpoint_path, checkpoint_manifest)
    terminal = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "accepted_at_audit",
        "selection_passed": True,
        "audit_opened": True,
        "audit_passed": True,
        "parameter_count": PARAMETER_COUNT,
        "widths": [row["widths"] for row in terminal_members],
        "depths": [row["depths"] for row in terminal_members],
        "ensemble_members": terminal_members,
        "ensemble_fusion": FUSION,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_predictions_read": False,
        "public_notebook_weights_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    write_json(target / "training_terminal.json", terminal)
    files = {
        path.relative_to(target).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(target.iterdir())
        if path.is_file()
    }
    write_json(
        target / "SOURCE_MANIFEST.json",
        {
            "schema_version": 1,
            "purpose": PURPOSE,
            "architecture": ARCHITECTURE_DESCRIPTION,
            "ensemble_fusion": FUSION,
            "parameter_count": PARAMETER_COUNT,
            "widths": terminal["widths"],
            "depths": terminal["depths"],
            "ensemble_size": len(terminal_members),
            "checkpoint_sha256": terminal["checkpoint_sha256"],
            "training_run_id": RUN_ID,
            "training_audit_passed": True,
            "competition_test_data_read_during_training": False,
            "public_notebook_weights_read_during_training": False,
            "public_leaderboard_used_for_selection": False,
            "member_evidence": terminal_members,
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": DATASET_TITLE,
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "checkpoint_sha256": terminal["checkpoint_sha256"],
                "parameter_count": PARAMETER_COUNT,
                "members": [row["name"] for row in terminal_members],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
