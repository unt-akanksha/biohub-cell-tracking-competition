#!/usr/bin/env python
"""Verify the immutable depth-robust conservative-PU detector archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import runpy


ROOT = Path(__file__).resolve().parents[1]
base = runpy.run_path(
    str(ROOT / "scripts" / "verify-antelume-peak-rank-detector-harvest.py")
)
verify = base["verify"]
verify.__globals__["ROOT"] = PurePosixPath(
    "synthetic256-real-conservative-pu-depth-robust-peak-rank-v2"
)
verify.__globals__["EXPECTED_RUN_ID"] = (
    "synthetic256-real-conservative-pu-depth-robust-peak-rank-v2"
)
verify.__globals__["EXPECTED_SEED"] = 1_407_733


def verify_depth_pu(archive_path: Path) -> dict:
    report = verify(archive_path)
    report.update(
        {
            "run_id": "antelume-peak-rank-depth-pu-v2-harvest-verification",
            "variant": "conservative_positive_unlabeled_depth_attenuation",
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_depth_pu(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
