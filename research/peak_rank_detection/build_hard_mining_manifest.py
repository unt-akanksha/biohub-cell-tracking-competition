#!/usr/bin/env python
"""Build a deterministic optimization-only hard-example sampling manifest.

Hardness is measured with the exact fixed temporal-minimum local-SNR response
that justified V23.  The manifest balances both embryo prefixes to a fixed
target count and assigns additional copies to the lowest-recall crops first.
Selection, sealed-audit, competition-test, and leaderboard data are never
opened.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.peak_rank_detection.diagnose_motion_supported_prior import (
    PRIOR_LOCAL_SNR_BANDS,
    reduced_local_snr_control,
)
from research.peak_rank_detection.diagnose_multiscale_blob_prior import (
    center_points,
    embryo_from_member_name,
    peak_coordinates,
    point_distances,
    sha256_file,
)


EXPECTED_UNIQUE_COUNTS = {"44b6": 150, "6bba": 330}
TARGET_EFFECTIVE_COUNT_PER_EMBRYO = 495


def assign_multiplicities(
    rows: Iterable[dict[str, Any]],
    *,
    expected_unique_counts: dict[str, int] = EXPECTED_UNIQUE_COUNTS,
    target_per_embryo: int = TARGET_EFFECTIVE_COUNT_PER_EMBRYO,
) -> list[dict[str, Any]]:
    """Repeat hard examples first until each embryo has the same count."""

    materialized = [dict(row) for row in rows]
    identities = [str(row["identity"]) for row in materialized]
    if len(identities) != len(set(identities)):
        raise ValueError("hard-mining identities must be unique")
    if target_per_embryo <= max(expected_unique_counts.values()):
        raise ValueError("target count must exceed every unique embryo count")

    output: list[dict[str, Any]] = []
    for embryo, expected_count in sorted(expected_unique_counts.items()):
        embryo_rows = [row for row in materialized if row.get("embryo") == embryo]
        if len(embryo_rows) != expected_count:
            raise ValueError(f"unexpected {embryo} optimization count")
        ranked = sorted(
            embryo_rows,
            key=lambda row: (
                float(row["top64_recall_at_2_5_voxels"]),
                -float(row["mean_capped_distance_voxels"]),
                str(row["identity"]),
            ),
        )
        multiplicities = {str(row["identity"]): 1 for row in ranked}
        remaining = target_per_embryo - len(ranked)
        while remaining:
            take = min(remaining, len(ranked))
            for row in ranked[:take]:
                multiplicities[str(row["identity"])] += 1
            remaining -= take
        for rank, row in enumerate(ranked):
            output.append(
                {
                    **row,
                    "hardness_rank_within_embryo": rank + 1,
                    "sampling_multiplicity": multiplicities[str(row["identity"])],
                }
            )
    return sorted(output, key=lambda row: str(row["identity"]))


def score_archive_member(item: tuple[str, bytes]) -> dict[str, Any]:
    name, content = item
    embryo = embryo_from_member_name(name)
    with np.load(io.BytesIO(content), allow_pickle=False) as payload:
        truth = center_points(payload["nodes"])
        response = reduced_local_snr_control(
            payload["volumes"], PRIOR_LOCAL_SNR_BANDS
        )
    distances = point_distances(
        peak_coordinates(response, maximum_predictions=64), truth
    )
    return {
        "identity": f"{Path(name).stem}:t1",
        "embryo": embryo,
        "annotated_points": int(len(truth)),
        "top64_recall_at_2_5_voxels": float(np.mean(distances <= 2.5)),
        "mean_capped_distance_voxels": float(np.mean(distances)),
        "p90_capped_distance_voxels": float(np.quantile(distances, 0.9)),
    }


def build_manifest(
    archive: Path, *, expected_sha256: str, workers: int = 1
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("expanded replay archive hash changed")
    with tarfile.open(archive, mode="r") as bundle:
        members = sorted(
            (
                member
                for member in bundle.getmembers()
                if member.isfile() and member.name.endswith(".npz")
            ),
            key=lambda member: member.name,
        )
        optimization = [member for member in members if "/optimization/" in member.name]
        if len(optimization) != sum(EXPECTED_UNIQUE_COUNTS.values()):
            raise ValueError("expanded replay optimization inventory changed")
        payloads: list[tuple[str, bytes]] = []
        for member in optimization:
            if "/selection/" in member.name or "/sealed_audit/" in member.name:
                raise ValueError("hard mining crossed the optimization boundary")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member: {member.name}")
            payloads.append((member.name, extracted.read()))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scored = list(executor.map(score_archive_member, payloads))
    entries = assign_multiplicities(scored)
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": "optimization-only-balanced-local-snr-hard-mining-v27",
        "archive_sha256": actual_sha256,
        "hardness_response": "v23_exact_temporal_minimum_local_snr_prior_bands",
        "local_snr_bands": [list(band) for band in PRIOR_LOCAL_SNR_BANDS],
        "ranking": [
            "ascending_top64_recall_at_2_5_voxels",
            "descending_mean_capped_distance_voxels",
            "ascending_identity",
        ],
        "maximum_predictions_per_crop": 64,
        "expected_unique_counts": EXPECTED_UNIQUE_COUNTS,
        "target_effective_count_per_embryo": TARGET_EFFECTIVE_COUNT_PER_EMBRYO,
        "effective_counts": {
            embryo: sum(
                int(row["sampling_multiplicity"])
                for row in entries
                if row["embryo"] == embryo
            )
            for embryo in sorted(EXPECTED_UNIQUE_COUNTS)
        },
        "entries": entries,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "selection_data_read": False,
        "sealed_audit_data_read": False,
        "public_predictions_read": False,
        "public_notebook_weights_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_manifest(
        args.archive.resolve(),
        expected_sha256=args.expected_sha256,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "effective_counts": report["effective_counts"],
                "entries": len(report["entries"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
