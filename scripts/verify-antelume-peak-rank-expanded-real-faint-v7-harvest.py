#!/usr/bin/env python
"""Verify the immutable expanded-real faint-cell detector archive."""

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
verify.__globals__.update(
    {
        "ROOT": PurePosixPath(
            "synthetic256-expanded-real-pu-faint-temporal-peak-rank-v7"
        ),
        "EXPECTED_RUN_ID": (
            "synthetic256-expanded-real-pu-faint-temporal-peak-rank-v7"
        ),
        "EXPECTED_PARAMETER_COUNT": 66_977_670,
        "EXPECTED_STEPS": 2_000,
        "EXPECTED_SEED": 4_709_011,
        "EXPECTED_WIDTHS": [128, 256, 512, 1024],
        "EXPECTED_DEPTHS": [3, 3, 9, 3],
    }
)


def verify_expanded_real_faint(archive_path: Path) -> dict:
    report = verify(archive_path)
    report.update(
        {
            "run_id": (
                "antelume-peak-rank-expanded-real-faint-v7-harvest-verification"
            ),
            "variant": (
                "capacity_conservative_pu_depth_temporal_fading_expanded_real"
            ),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_expanded_real_faint(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
