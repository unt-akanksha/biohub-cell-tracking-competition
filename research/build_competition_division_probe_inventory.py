#!/usr/bin/env python
"""Build a competition-train division probe without selecting a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.division_recovery_feasibility import audit_movie, graph_plain
from research.learned_division_recovery import (
    VOXEL_SIZE_ZYX_UM,
    discover_division_recovery_candidates,
)


def candidate_geometry_features(
    nodes: dict[int, dict[str, float | int]],
    incoming: dict[int, list[int]],
    *,
    parent_id: int,
    existing_child_id: int,
    second_child_id: int,
) -> dict[str, float | int | None]:
    def physical(node_id: int) -> np.ndarray:
        node = nodes[node_id]
        return np.asarray((node["z"], node["y"], node["x"]), dtype=np.float64) * VOXEL_SIZE_ZYX_UM

    parent = physical(parent_id)
    existing_step = physical(existing_child_id) - parent
    second_step = physical(second_child_id) - parent
    existing_distance = float(np.linalg.norm(existing_step))
    second_distance = float(np.linalg.norm(second_step))
    midpoint = parent + 0.5 * (existing_step + second_step)
    denominator = max(existing_distance * second_distance, 1e-12)
    opposition_cosine = float(np.dot(existing_step, second_step) / denominator)
    predecessor_ids = [
        node_id
        for node_id in incoming.get(parent_id, [])
        if int(nodes[node_id]["t"]) == int(nodes[parent_id]["t"]) - 1
    ]
    predecessor_id = predecessor_ids[0] if len(predecessor_ids) == 1 else None
    velocity = parent - physical(predecessor_id) if predecessor_id is not None else None
    return {
        "existing_distance_um": existing_distance,
        "daughter_midpoint_distance_um": float(np.linalg.norm(midpoint - parent)),
        "daughter_opposition_cosine": opposition_cosine,
        "daughter_step_ratio": float(
            min(existing_distance, second_distance)
            / max(existing_distance, second_distance, 1e-12)
        ),
        "predecessor_id": predecessor_id,
        "parent_velocity_um": float(np.linalg.norm(velocity)) if velocity is not None else None,
        "constant_velocity_midpoint_error_um": (
            float(np.linalg.norm(midpoint - (parent + velocity)))
            if velocity is not None
            else None
        ),
    }


def movie_inventory(prediction_path: Path, truth_path: Path) -> dict:
    audit = audit_movie(prediction_path, truth_path)
    plain_nodes, plain_edges = graph_plain(prediction_path)
    nodes = {
        node_id: {
            "t": int(row[0]),
            "z": float(row[1]),
            "y": float(row[2]),
            "x": float(row[3]),
        }
        for node_id, row in plain_nodes.items()
    }
    edges = [
        {"source_id": int(source), "target_id": int(target)}
        for source, target in plain_edges
    ]
    incoming: dict[int, list[int]] = {}
    for source, target in plain_edges:
        incoming.setdefault(int(target), []).append(int(source))
    candidates = discover_division_recovery_candidates(nodes, edges)
    event_times = {int(row["timepoint"]) for row in audit["division_events"]}
    safe_positive_parents = {
        int(row["mapped_prediction_ids_7um"][0])
        for row in audit["division_events"]
        if row.get("recoverable_single_missing_daughter") is True
        and row.get("missing_daughter", {}).get("requires_parent_reassignment") is False
    }
    rows = []
    for candidate in candidates:
        node = nodes[candidate.parent_id]
        if int(node["t"]) not in event_times:
            continue
        rows.append(
            {
                "parent_id": candidate.parent_id,
                "timepoint": int(node["t"]),
                "center_zyx_voxel": [node["z"], node["y"], node["x"]],
                "existing_child_id": candidate.existing_child_id,
                "second_child_id": candidate.second_child_id,
                "parent_distance_um": candidate.parent_distance_um,
                "sister_distance_um": candidate.sister_distance_um,
                **candidate_geometry_features(
                    nodes,
                    incoming,
                    parent_id=candidate.parent_id,
                    existing_child_id=candidate.existing_child_id,
                    second_child_id=candidate.second_child_id,
                ),
                "safe_recovery_positive": candidate.parent_id in safe_positive_parents,
            }
        )
    found = {
        int(row["parent_id"])
        for row in rows
        if row["safe_recovery_positive"] is True
    }
    if found != safe_positive_parents:
        raise RuntimeError(
            f"safe competition division candidates changed for {truth_path.stem}: "
            f"expected={sorted(safe_positive_parents)}, found={sorted(found)}"
        )
    frames = sorted(
        {
            max(0, timepoint + offset)
            for timepoint in event_times
            for offset in (-1, 0, 1)
        }
    )
    return {
        "stem": truth_path.stem,
        "event_timepoints": sorted(event_times),
        "required_frames": frames,
        "safe_positive_parents": sorted(safe_positive_parents),
        "candidates": rows,
    }


def find_graph(root: Path, stem: str) -> Path:
    matches = sorted(path for path in root.glob(f"{stem}.geff") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"expected one prediction graph for {stem}, saw {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth_paths = sorted(args.truth_root.glob("*.geff"))
    if not truth_paths:
        raise FileNotFoundError("competition-train truth GEFFs are missing")
    movies = [
        movie_inventory(find_graph(args.prediction_root, path.stem), path)
        for path in truth_paths
    ]
    positives = sum(
        bool(row["safe_recovery_positive"])
        for movie in movies
        for row in movie["candidates"]
    )
    result = {
        "schema_version": 1,
        "status": "competition_train_probe_only",
        "movies": movies,
        "summary": {
            "movies": len(movies),
            "event_frames": sum(len(movie["event_timepoints"]) for movie in movies),
            "geometric_candidates": sum(len(movie["candidates"]) for movie in movies),
            "safe_recovery_positives": positives,
        },
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    if positives != 3:
        raise RuntimeError(f"expected three parent-free recoverable events, saw {positives}")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
