#!/usr/bin/env python
"""Verify the immutable V27 hard-mined temporal-SNR detector archive."""

from __future__ import annotations

import argparse
import json
import runpy
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PurePosixPath(
    "synthetic256-expanded-real-temporal-min-balanced-hard-mined-pu-faint-"
    "local-shape-multiscale-blob-global-safe-rank-peak-rank-v27"
)
SAMPLING_MANIFEST_SHA256 = (
    "9967efa25021453b4f043a23e47e744153da678f17f5c00180ecd6ebd4dca900"
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
        "EXPECTED_STEPS": 6_000,
        "EXPECTED_SEED": 13_679_443,
        "EXPECTED_WIDTHS": [128, 256, 512, 1024],
        "EXPECTED_DEPTHS": [3, 3, 9, 3],
        "EXPECTED_MODEL_FAMILY": (
            "temporal_min_local_snr_safe_rank_multiscale_global_peak_rank_v23"
        ),
    }
)


def verify_hard_mined_temporal_snr(archive_path: Path) -> dict:
    report = verify(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember((ARCHIVE_ROOT / "terminal.json").as_posix())
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("V27 terminal is unreadable")
        terminal = json.loads(stream.read())
    receipt = terminal.get("hard_mining_receipt", {})
    if not (
        terminal.get("safe_negative_evidence_band") == [5, 13]
        and terminal.get("safe_negative_boundary_quantile") == 0.5
        and terminal.get("hard_mining_role") == "expanded_real_optimization_only"
        and terminal.get("real_optimization_sampling_policy")
        == "embryo_balanced_495_each_v23_local_snr_hardest_first"
        and terminal.get("hard_mining_selection_sampling_changed") is False
        and terminal.get("hard_mining_sealed_audit_sampling_changed") is False
        and receipt.get("manifest_sha256") == SAMPLING_MANIFEST_SHA256
        and receipt.get("manifest_run_id")
        == "optimization-only-balanced-local-snr-hard-mining-v27"
        and receipt.get("unique_counts") == {"44b6": 150, "6bba": 330}
        and receipt.get("effective_counts") == {"44b6": 495, "6bba": 495}
        and receipt.get("effective_examples") == 990
        and receipt.get("multiplicity_histogram")
        == {"44b6": {"3": 105, "4": 45}, "6bba": {"1": 165, "2": 165}}
    ):
        raise ValueError("V27 hard-mining contract changed")
    report.update(
        {
            "run_id": "antelume-peak-rank-hard-mined-temporal-snr-v27-verification",
            "variant": "balanced_hard_mined_temporal_minimum_local_snr_safe_rank",
            "optimization_sampling_policy": (
                "embryo_balanced_495_each_v23_local_snr_hardest_first"
            ),
            "hard_mining_manifest_sha256": SAMPLING_MANIFEST_SHA256,
            "effective_optimization_counts": {"44b6": 495, "6bba": 495},
            "temporal_reducer": "minimum",
            "local_snr_bands": [[3, 9], [3, 11], [5, 13]],
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_hard_mined_temporal_snr(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
