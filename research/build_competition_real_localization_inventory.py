#!/usr/bin/env python
"""Freeze train-only real Biohub triplets for localization-domain replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.build_competition_division_training_inventory import (
    FINAL_PROBE_STEMS,
    sha256_file,
    validate_geff_cache,
)
from research.division_recovery_feasibility import graph_plain


RUN_ID = "competition-real-localization-inventory-v1"
SOURCE_RUN_ID = "competition-real-division-training-inventory-v1"
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "93ec74335e2c80dbfe773de0cb13e99ad0e96e2c6a3aade155971305cb6e48f6"
)
NONDIVISION_MOVIES_PER_EMBRYO = 16


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def stable_rank(stem: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{stem}".encode("utf-8")).hexdigest()


def split_selection_roles(movies: list[dict[str, Any]]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for embryo in ("44b6", "6bba"):
        selected = sorted(
            (
                movie
                for movie in movies
                if movie["embryo"] == embryo and movie["role"] == "selection"
            ),
            key=lambda movie: stable_rank(movie["stem"], "real-localization-audit-v1"),
        )
        if len(selected) < 2:
            raise ValueError(f"real localization split lacks {embryo} selection movies")
        audit_count = len(selected) // 2
        for movie in selected[:audit_count]:
            roles[movie["stem"]] = "sealed_audit"
        for movie in selected[audit_count:]:
            roles[movie["stem"]] = "selection"
    return roles


def selected_nondivision_stems(movies: list[dict[str, Any]]) -> set[str]:
    selected: set[str] = set()
    for embryo in ("44b6", "6bba"):
        eligible = sorted(
            (
                movie
                for movie in movies
                if movie["embryo"] == embryo
                and movie["role"] == "optimization"
                and not movie["event_timepoints"]
            ),
            key=lambda movie: stable_rank(
                movie["stem"], "real-localization-nondivision-v1"
            ),
        )
        if len(eligible) < NONDIVISION_MOVIES_PER_EMBRYO:
            raise ValueError(f"not enough {embryo} non-division optimization movies")
        selected.update(
            movie["stem"] for movie in eligible[:NONDIVISION_MOVIES_PER_EMBRYO]
        )
    return selected


def fallback_center(stem: str, nodes: dict[int, tuple[float, ...]]) -> int:
    maximum_time = max(int(values[0]) for values in nodes.values())
    eligible = sorted(
        {
            int(values[0])
            for values in nodes.values()
            if 0 < int(values[0]) < maximum_time
        }
    )
    if not eligible:
        raise ValueError(f"real localization movie has no interior labels: {stem}")
    digest = stable_rank(stem, "real-localization-center-v1")
    return eligible[int(digest[:16], 16) % len(eligible)]


def validate_source(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    movies = payload.get("movies")
    if not (
        sha256_file(path) == EXPECTED_SOURCE_INVENTORY_SHA256
        and payload.get("schema_version") == 1
        and payload.get("status") == "complete"
        and payload.get("run_id") == SOURCE_RUN_ID
        and payload.get("final_probe_stems") == sorted(FINAL_PROBE_STEMS)
        and payload.get("competition_train_data_read") is True
        and payload.get("competition_test_data_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and payload.get("authorized_for_submission") is False
        and isinstance(movies, list)
        and len(movies) == 199
    ):
        raise ValueError("real localization source inventory changed")
    return movies


def build_inventory(
    movies: list[dict[str, Any]], geff_root: Path
) -> list[dict[str, Any]]:
    selection_roles = split_selection_roles(movies)
    nondivision = selected_nondivision_stems(movies)
    records: list[dict[str, Any]] = []
    for movie in movies:
        stem = str(movie["stem"])
        if stem in FINAL_PROBE_STEMS:
            continue
        nodes, _edges = graph_plain(geff_root / "train" / f"{stem}.geff")
        maximum_time = max(int(values[0]) for values in nodes.values())
        valid_events = sorted(
            {
                int(value)
                for value in movie["event_timepoints"]
                if 0 < int(value) < maximum_time
            }
        )
        original_role = str(movie["role"])
        if original_role == "selection":
            role = selection_roles[stem]
            centers = valid_events
        elif original_role == "optimization" and valid_events:
            role = "optimization"
            centers = valid_events
        elif original_role == "optimization" and stem in nondivision:
            role = "optimization"
            centers = []
        else:
            continue
        if not centers:
            if original_role == "selection":
                raise ValueError(f"real localization selection lost interior events: {stem}")
            centers = [fallback_center(stem, nodes)]
        if any(center <= 0 or center >= maximum_time for center in centers):
            raise ValueError(f"real localization center is not interior: {stem}")
        annotated = sum(
            int(values[0]) in centers for values in nodes.values()
        )
        if annotated <= 0:
            raise ValueError(f"real localization center has no labels: {stem}")
        records.append(
            {
                "stem": stem,
                "embryo": str(movie["embryo"]),
                "role": role,
                "center_frames": sorted(set(centers)),
                "required_frames": sorted(
                    {
                        center + offset
                        for center in centers
                        for offset in (-1, 0, 1)
                    }
                ),
                "annotated_nodes_at_centers": annotated,
                "division_centers": len(valid_events),
            }
        )
    return sorted(records, key=lambda row: (row["role"], row["stem"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--geff-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_payload = json.loads(args.source_inventory.read_text(encoding="utf-8"))
    movies = validate_source(source_payload, args.source_inventory)
    geff_manifest_path = args.geff_cache_root / "train_geff_cache_manifest.json"
    geff_manifest = json.loads(geff_manifest_path.read_text(encoding="utf-8"))
    validate_geff_cache(args.geff_cache_root, geff_manifest)
    records = build_inventory(movies, args.geff_cache_root)
    counts = {
        role: {
            "movies": sum(row["role"] == role for row in records),
            "center_frames": sum(
                len(row["center_frames"]) for row in records if row["role"] == role
            ),
            "required_frames": sum(
                len(row["required_frames"])
                for row in records
                if row["role"] == role
            ),
            "annotated_nodes_at_centers": sum(
                row["annotated_nodes_at_centers"]
                for row in records
                if row["role"] == role
            ),
        }
        for role in ("optimization", "selection", "sealed_audit")
    }
    if (
        counts["optimization"]["movies"] != 96
        or counts["selection"]["movies"] != 10
        or counts["sealed_audit"]["movies"] != 9
        or set(row["stem"] for row in records) & FINAL_PROBE_STEMS
        or any(counts[role]["annotated_nodes_at_centers"] <= 0 for role in counts)
    ):
        raise RuntimeError("real localization role inventory changed")
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "source_inventory_sha256": sha256_file(args.source_inventory),
        "source_geff_manifest_sha256": sha256_file(geff_manifest_path),
        "split_policy": (
            "all division-positive optimization movies plus sixteen stable-hash "
            "non-division movies per embryo; prior selection movies split by "
            "stable hash within embryo into real selection and sealed audit"
        ),
        "movies": records,
        "summary": {
            "movies": len(records),
            "center_frames": sum(len(row["center_frames"]) for row in records),
            "required_frames": sum(len(row["required_frames"]) for row in records),
            "annotated_nodes_at_centers": sum(
                row["annotated_nodes_at_centers"] for row in records
            ),
            "by_role": counts,
        },
        "excluded_final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
