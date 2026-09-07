#!/usr/bin/env python
"""Verify the immutable capacity-scaled conservative-PU detector archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
import runpy


ROOT = Path(__file__).resolve().parents[1]
base = runpy.run_path(
    str(ROOT / "scripts" / "verify-antelume-peak-rank-detector-harvest.py")
)
verify = base["verify"]
verify.__globals__.update(
    {
        "ROOT": PurePosixPath(
            "synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3"
        ),
        "EXPECTED_RUN_ID": (
            "synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3"
        ),
        "EXPECTED_PARAMETER_COUNT": 66_977_670,
        "EXPECTED_STEPS": 2_000,
        "EXPECTED_SEED": 2_607_157,
        "EXPECTED_WIDTHS": [128, 256, 512, 1024],
        "EXPECTED_DEPTHS": [3, 3, 9, 3],
    }
)


def verify_capacity_pu(archive_path: Path) -> dict:
    report = verify(archive_path)
    if report["accepted_for_kaggle_validation"]:
        root = (
            "synthetic256-real-conservative-pu-depth-robust-capacity-peak-rank-v3"
        )
        with tarfile.open(archive_path, "r:gz") as archive:
            calibration_bytes = archive.extractfile(
                f"{root}/threshold_calibration.json"
            ).read()
            terminal_bytes = archive.extractfile(f"{root}/terminal.json").read()
        calibration = json.loads(calibration_bytes)
        thresholds = calibration.get("thresholds")
        views = {"none": 1, "zflip2": 2, "rot4": 4, "d4": 8}
        if not (
            calibration.get("run_id")
            == "synthetic-complete-global-peak-threshold-v1"
            and calibration.get("status") == "calibrated"
            and calibration.get("threshold_policy")
            == "synthetic_selection_micro_detection_jaccard"
            and calibration.get("checkpoint_sha256")
            == report["checkpoint_sha256"]
            and calibration.get("training_terminal_sha256")
            == hashlib.sha256(terminal_bytes).hexdigest()
            and calibration.get("parameter_count") == 66_977_670
            and calibration.get("selection_indices") == list(range(240, 248))
            and calibration.get("complete_synthetic_labels_read") is True
            and calibration.get("competition_train_data_read") is False
            and calibration.get("competition_test_data_read") is False
            and calibration.get("organizer_estimated_node_count_read") is False
            and calibration.get(
                "organizer_estimated_node_count_used_for_threshold"
            )
            is False
            and calibration.get("public_predictions_read") is False
            and calibration.get("public_notebook_weights_read") is False
            and calibration.get("public_leaderboard_used_for_selection") is False
            and isinstance(thresholds, dict)
            and set(thresholds) == set(views)
            and all(
                isinstance(thresholds[mode].get("threshold"), (int, float))
                and not isinstance(thresholds[mode].get("threshold"), bool)
                and 0.0 < float(thresholds[mode]["threshold"]) < 1.0
                and thresholds[mode].get("tta_mode") == mode
                and thresholds[mode].get("tta_views") == count
                and thresholds[mode].get("complete_synthetic_examples") == 24
                for mode, count in views.items()
            )
        ):
            raise ValueError("capacity V3 clean threshold calibration is invalid")
        report["threshold_calibration_sha256"] = hashlib.sha256(
            calibration_bytes
        ).hexdigest()
        report["peak_threshold_policy"] = calibration["threshold_policy"]
        report["organizer_estimated_node_count_used_for_threshold"] = False
    report.update(
        {
            "run_id": "antelume-peak-rank-capacity-pu-v3-harvest-verification",
            "variant": "capacity_scaled_conservative_pu_depth_attenuation",
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_capacity_pu(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
