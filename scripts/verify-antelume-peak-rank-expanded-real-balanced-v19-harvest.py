#!/usr/bin/env python
"""Verify the immutable embryo-balanced safe-rank detector archive."""

from __future__ import annotations

import argparse
import json
import runpy
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PurePosixPath(
    "synthetic256-expanded-real-balanced-pu-faint-local-shape-multiscale-"
    "blob-global-safe-rank-peak-rank-v19"
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
        "EXPECTED_STEPS": 5_000,
        "EXPECTED_SEED": 10_346_297,
        "EXPECTED_WIDTHS": [128, 256, 512, 1024],
        "EXPECTED_DEPTHS": [3, 3, 9, 3],
        "EXPECTED_MODEL_FAMILY": (
            "safe_rank_multiscale_blob_global_temporal_peak_rank_v17"
        ),
    }
)


def verify_balanced_safe_rank(archive_path: Path) -> dict:
    report = verify(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember((ARCHIVE_ROOT / "terminal.json").as_posix())
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("balanced safe-rank terminal is unreadable")
        terminal = json.loads(stream.read())
    if not (
        terminal.get("safe_negative_evidence_band") == [5, 13]
        and terminal.get("safe_negative_boundary_quantile") == 0.5
        and terminal.get("safe_negative_shell_radius_voxels") == [3.0, 8.0]
        and terminal.get("safe_negative_selection_role")
        == "expanded_real_optimization_only"
        and terminal.get("real_optimization_sampling_policy")
        == "duplicate_44b6_once_balance_embryo_crops"
        and terminal.get("real_optimization_original_counts")
        == {"44b6": 150, "6bba": 330}
        and terminal.get("real_optimization_effective_counts")
        == {"44b6": 300, "6bba": 330}
        and terminal.get("selection_sampling_changed") is False
        and terminal.get("sealed_audit_sampling_changed") is False
    ):
        raise ValueError("balanced safe-rank contract changed")
    report.update(
        {
            "run_id": (
                "antelume-peak-rank-expanded-real-balanced-v19-harvest-"
                "verification"
            ),
            "variant": "capacity_expanded_real_balanced_safe_rank",
            "optimization_sampling_policy": (
                "duplicate_44b6_once_balance_embryo_crops"
            ),
            "optimization_original_counts": {"44b6": 150, "6bba": 330},
            "optimization_effective_counts": {"44b6": 300, "6bba": 330},
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_balanced_safe_rank(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
