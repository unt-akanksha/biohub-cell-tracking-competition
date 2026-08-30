#!/usr/bin/env python
"""Build reciprocal real-Biohub division training inventories from train GEFFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.division_recovery_feasibility import graph_plain


RUN_ID = "competition-real-division-training-inventory-v1"
FINAL_PROBE_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_geff_cache(root: Path, manifest: dict[str, Any]) -> list[str]:
    summary = manifest.get("summary", {})
    stems = manifest.get("stems")
    files = manifest.get("files")
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == "competition-train-geff-cache-v1"
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("authorized_for_submission") is False
        and isinstance(stems, list)
        and len(stems) == 199
        and isinstance(files, list)
        and len(files) >= 3_000
        and summary.get("movies") == 199
        and summary.get("members") == len(files)
        and summary.get("embryos") == {"44b6": 71, "6bba": 128}
    ):
        raise ValueError("competition train GEFF cache is ineligible")
    for record in files:
        relative = str(record["relative_path"])
        path = root / "train" / relative
        if (
            not relative.endswith(("zarr.json", "/0", "/0/0"))
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"competition train GEFF cache changed: {relative}")
    return [str(stem) for stem in stems]


def movie_examples(
    nodes: dict[int, tuple[float, ...]], edges: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    outgoing: dict[int, set[int]] = {}
    for source, target in edges:
        if source not in nodes or target not in nodes:
            raise ValueError("train GEFF contains a dangling edge")
        if int(nodes[target][0]) != int(nodes[source][0]) + 1:
            raise ValueError("train GEFF contains a non-adjacent edge")
        outgoing.setdefault(int(source), set()).add(int(target))
    division_times = {
        int(nodes[node_id][0])
        for node_id, targets in outgoing.items()
        if len(targets) >= 2
    }
    rows: list[dict[str, Any]] = []
    for node_id, targets in sorted(outgoing.items()):
        timepoint = int(nodes[node_id][0])
        if timepoint not in division_times or len(targets) not in (1, 2):
            continue
        rows.append(
            {
                "node_id": int(node_id),
                "timepoint": timepoint,
                "center_zyx_voxel": [float(value) for value in nodes[node_id][1:]],
                "division_target": len(targets) == 2,
                "child_ids": sorted(int(target) for target in targets),
            }
        )
    return rows


def stratified_selection_stems(movies: list[dict[str, Any]]) -> set[str]:
    selected: set[str] = set()
    for embryo in ("44b6", "6bba"):
        eligible = [
            movie
            for movie in movies
            if movie["embryo"] == embryo
            and movie["role"] != "final_probe"
            and movie["summary"]["division_positives"] > 0
        ]
        target = int(
            math.ceil(
                sum(movie["summary"]["division_positives"] for movie in eligible)
                * 0.20
            )
        )
        accumulated = 0
        for movie in sorted(
            eligible,
            key=lambda row: hashlib.sha256(row["stem"].encode("utf-8")).hexdigest(),
        ):
            selected.add(movie["stem"])
            accumulated += int(movie["summary"]["division_positives"])
            if accumulated >= target:
                break
        if accumulated < target:
            raise RuntimeError(f"division-stratified selection failed for {embryo}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.cache_root / "train_geff_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stems = validate_geff_cache(args.cache_root, manifest)
    movies: list[dict[str, Any]] = []
    for index, stem in enumerate(stems, start=1):
        nodes, edges = graph_plain(args.cache_root / "train" / f"{stem}.geff")
        examples = movie_examples(nodes, edges)
        role = "final_probe" if stem in FINAL_PROBE_STEMS else "unassigned"
        event_timepoints = sorted(
            {row["timepoint"] for row in examples if row["division_target"]}
        )
        movies.append(
            {
                "stem": stem,
                "embryo": stem.split("_", 1)[0],
                "role": role,
                "event_timepoints": event_timepoints,
                "required_frames": sorted(
                    {
                        timepoint + offset
                        for timepoint in event_timepoints
                        for offset in (-1, 0, 1)
                    }
                ),
                "examples": examples,
                "summary": {
                    "division_positives": sum(
                        row["division_target"] for row in examples
                    ),
                    "same_frame_ordinary_negatives": sum(
                        not row["division_target"] for row in examples
                    ),
                },
            }
        )
        if index % 25 == 0:
            print(f"read {index}/{len(stems)} train GEFFs", flush=True)
    selection_stems = stratified_selection_stems(movies)
    for movie in movies:
        if movie["role"] == "unassigned":
            movie["role"] = (
                "selection" if movie["stem"] in selection_stems else "optimization"
            )
    usable = [movie for movie in movies if movie["role"] != "final_probe"]
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for embryo in ("44b6", "6bba"):
        counts[embryo] = {}
        for role in ("optimization", "selection", "final_probe"):
            selected = [
                movie
                for movie in movies
                if movie["embryo"] == embryo and movie["role"] == role
            ]
            counts[embryo][role] = {
                "movies": len(selected),
                "event_frames": sum(len(movie["event_timepoints"]) for movie in selected),
                "division_positives": sum(
                    movie["summary"]["division_positives"] for movie in selected
                ),
                "same_frame_ordinary_negatives": sum(
                    movie["summary"]["same_frame_ordinary_negatives"]
                    for movie in selected
                ),
            }
    if set(movie["stem"] for movie in movies if movie["role"] == "final_probe") != FINAL_PROBE_STEMS:
        raise RuntimeError("final four-movie division probe exclusion changed")
    for embryo in ("44b6", "6bba"):
        if any(
            counts[embryo][role]["division_positives"] <= 0
            for role in ("optimization", "selection", "final_probe")
        ):
            raise RuntimeError(f"reciprocal division split lost positives for {embryo}")
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "split_policy": (
            "final four complete movies excluded; division-positive movies "
            "ordered by sha256(stem) independently within each embryo until "
            "at least 20 percent of division positives enter selection"
        ),
        "movies": movies,
        "summary": {
            "movies": len(movies),
            "training_movies": len(usable),
            "division_positives": sum(
                movie["summary"]["division_positives"] for movie in usable
            ),
            "same_frame_ordinary_negatives": sum(
                movie["summary"]["same_frame_ordinary_negatives"] for movie in usable
            ),
            "required_train_frames": sum(
                len(movie["required_frames"]) for movie in usable
            ),
            "by_embryo_role": counts,
        },
        "final_probe_stems": sorted(FINAL_PROBE_STEMS),
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
