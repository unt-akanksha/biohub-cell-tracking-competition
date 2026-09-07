#!/usr/bin/env python
"""Bind a clean Kaggle promotion result into the private peak runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / ".biohub" / "automation" / "peak-rank-validation-controller-v1.json"
RUNTIME = ROOT / ".biohub" / "staging" / "biohub-peak-rank-validation-runtime-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def locate_result(controller: dict) -> Path:
    root = Path(controller["download_root"])
    matches = list(root.rglob("peak_rank_validation.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one clean validation result, found {matches}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=CONTROLLER)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    if not (
        controller.get("status") == "completed"
        and controller.get("promotion_passed") is True
        and controller.get("accepted_for_candidate_integration") is True
        and controller.get("competition_submission_performed") is False
    ):
        raise RuntimeError("peak-ranking clean validation was not promoted")
    result_path = locate_result(controller)
    result_hash = sha256_file(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        result_hash == controller.get("validation_result_sha256")
        and result.get("run_id") == "temporal-peak-rank-clean-validation-v1"
        and result.get("selection_passed") is True
        and result.get("acceptance_opened") is True
        and result.get("promotion_passed") is True
        and result.get("selected_tta_mode") in {"none", "zflip2", "rot4", "d4"}
        and result.get("selected_tta_views") in {1, 2, 4, 8}
        and isinstance(result.get("selected_peak_threshold"), (int, float))
        and not isinstance(result.get("selected_peak_threshold"), bool)
        and 0.0 < float(result["selected_peak_threshold"]) < 1.0
        and result.get("threshold_policy")
        == "synthetic_selection_micro_detection_jaccard"
        and result.get("organizer_estimated_node_count_read") is False
        and result.get("organizer_estimated_node_count_used_for_threshold") is False
        and result.get("competition_test_data_read") is False
        and result.get("competition_submission_performed") is False
    ):
        raise RuntimeError("peak-ranking promotion evidence is inconsistent")
    manifest_path = args.runtime / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(args.runtime / "peak_rank_detector.pt")
    threshold_calibration = args.runtime / "threshold_calibration.json"
    if not (
        manifest.get("training_audit_passed") is True
        and manifest.get("checkpoint_sha256") == checkpoint_hash
        and controller.get("checkpoint_sha256") == checkpoint_hash
        and result.get("provenance", {}).get("checkpoint_sha256") == checkpoint_hash
        and threshold_calibration.is_file()
        and manifest.get("threshold_calibration_sha256")
        == sha256_file(threshold_calibration)
        and result.get("threshold_calibration_sha256")
        == manifest.get("threshold_calibration_sha256")
        and manifest.get("peak_threshold_policy")
        == "synthetic_selection_micro_detection_jaccard"
        and manifest.get("organizer_estimated_node_count_read") is False
        and manifest.get("organizer_estimated_node_count_used_for_threshold") is False
    ):
        raise RuntimeError("clean validation is bound to another checkpoint")
    destination = args.runtime / "clean_validation.json"
    if destination.exists() and not args.replace:
        raise FileExistsError(destination)
    shutil.copy2(result_path, destination)
    manifest["purpose"] = (
        "Clean-promoted temporal peak detector, official-linker production bridge, "
        "and two-GPU validation"
    )
    manifest["clean_validation_promotion_passed"] = True
    manifest["clean_validation_sha256"] = result_hash
    manifest["selected_peak_tta_mode"] = result["selected_tta_mode"]
    manifest["selected_peak_tta_views"] = result["selected_tta_views"]
    manifest["selected_peak_threshold"] = result["selected_peak_threshold"]
    manifest["files"][destination.name] = {
        "bytes": destination.stat().st_size,
        "sha256": result_hash,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "runtime": str(args.runtime),
                "checkpoint_sha256": checkpoint_hash,
                "clean_validation_sha256": result_hash,
                "selected_peak_tta_mode": result["selected_tta_mode"],
                "selected_peak_threshold": result["selected_peak_threshold"],
                "requires_private_dataset_version": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
