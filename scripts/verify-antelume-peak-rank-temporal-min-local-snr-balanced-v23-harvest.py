#!/usr/bin/env python
"""Verify the immutable temporal-minimum local-SNR detector archive."""

from __future__ import annotations

import argparse
import json
import runpy
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PurePosixPath(
    "synthetic256-expanded-real-temporal-min-balanced-pu-faint-local-shape-"
    "multiscale-blob-global-safe-rank-peak-rank-v23"
)
base = runpy.run_path(
    str(ROOT / "scripts" / "verify-antelume-peak-rank-detector-harvest.py")
)
verify = base["verify"]
verify.__globals__.update(
    {
        "ROOT": ARCHIVE_ROOT,
        "EXPECTED_RUN_ID": str(ARCHIVE_ROOT),
        "EXPECTED_PARAMETER_COUNT": 83_812_614,
        "EXPECTED_STEPS": 5_000,
        "EXPECTED_SEED": 12_568_331,
        "EXPECTED_WIDTHS": [128, 256, 512, 1024],
        "EXPECTED_DEPTHS": [3, 3, 9, 3],
        "EXPECTED_MODEL_FAMILY": (
            "temporal_min_local_snr_safe_rank_multiscale_global_peak_rank_v23"
        ),
    }
)


def verify_temporal_minimum_local_snr(archive_path: Path) -> dict:
    report = verify(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember((ARCHIVE_ROOT / "terminal.json").as_posix())
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("temporal-minimum local-SNR terminal is unreadable")
        terminal = json.loads(stream.read())
    if not (
        terminal.get("safe_negative_evidence_band") == [5, 13]
        and terminal.get("safe_negative_boundary_quantile") == 0.5
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
        raise ValueError("temporal-minimum local-SNR safe-rank contract changed")
    report.update(
        {
            "run_id": (
                "antelume-peak-rank-temporal-min-local-snr-balanced-v23-"
                "harvest-verification"
            ),
            "variant": "temporal_minimum_local_snr_balanced_safe_rank",
            "optimization_sampling_policy": (
                "duplicate_44b6_once_balance_embryo_crops"
            ),
            "temporal_reducer": "minimum",
            "local_snr_bands": [[3, 9], [5, 13], [7, 15]],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_temporal_minimum_local_snr(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
