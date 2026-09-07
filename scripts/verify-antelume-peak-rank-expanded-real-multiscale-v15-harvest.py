#!/usr/bin/env python
"""Verify the immutable expanded-real multiscale detector archive."""

from __future__ import annotations

import argparse
import json
import runpy
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PurePosixPath(
    "synthetic256-expanded-real-pu-faint-local-shape-multiscale-blob-global-"
    "peak-rank-v15"
)
base = runpy.run_path(
    str(ROOT / "scripts" / "verify-antelume-peak-rank-detector-harvest.py")
)
verify = base["verify"]
verify.__globals__.update(
    {
        "ROOT": ARCHIVE_ROOT,
        "EXPECTED_RUN_ID": str(ARCHIVE_ROOT),
        "EXPECTED_PARAMETER_COUNT": 83_802_246,
        "EXPECTED_STEPS": 3_000,
        "EXPECTED_SEED": 8_124_071,
        "EXPECTED_WIDTHS": [128, 256, 512, 1024],
        "EXPECTED_DEPTHS": [3, 3, 9, 3],
        "EXPECTED_MODEL_FAMILY": (
            "multiscale_blob_global_context_temporal_peak_rank_v15"
        ),
    }
)


def verify_multiscale(archive_path: Path) -> dict:
    report = verify(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember((ARCHIVE_ROOT / "terminal.json").as_posix())
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("multiscale terminal is unreadable")
        terminal = json.loads(stream.read())
    if not (
        terminal.get("blob_bands") == [[3, 9], [5, 13], [7, 15]]
        and terminal.get("blob_scale_selection_role")
        == "expanded_real_optimization_only"
    ):
        raise ValueError("multiscale scale-space contract changed")
    report.update(
        {
            "run_id": (
                "antelume-peak-rank-expanded-real-multiscale-v15-harvest-"
                "verification"
            ),
            "variant": (
                "capacity_expanded_real_faint_local_shape_multiscale_blob_"
                "global_context"
            ),
            "blob_bands": terminal["blob_bands"],
            "blob_scale_selection_role": terminal["blob_scale_selection_role"],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_multiscale(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
