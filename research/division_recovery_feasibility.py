#!/usr/bin/env python
"""Audit whether missed divisions are recoverable from already detected nodes.

This is an oracle feasibility diagnostic, not a threshold-selection tool. It
uses ground truth only to describe missed events and the rank of the known
second daughter among parent-free detections in the next frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.trackastra_graph.train_biohub_graph_transformer import (
    VOXEL_SCALE_UM,
    compute_division_confusion,
    compute_edge_confusion,
    match_nodes_bipartite,
    read_graph_video,
)


def graph_plain(path: Path) -> tuple[dict[int, tuple[float, ...]], list[tuple[int, int]]]:
    video = read_graph_video(path)
    nodes = {
        int(node_id): (int(timepoint), *(float(value) for value in coordinate))
        for node_id, timepoint, coordinate in zip(
            video.node_ids, video.times, video.coords_voxel, strict=True
        )
    }
    edges = [tuple(int(value) for value in edge) for edge in video.edges]
    return nodes, edges


def ranked_second_daughter_candidates(
    nodes: dict[int, tuple[float, ...]],
    edges: list[tuple[int, int]],
    *,
    parent_id: int,
    existing_child_id: int,
) -> list[dict[str, Any]]:
    if parent_id not in nodes or existing_child_id not in nodes:
        raise KeyError("division parent and existing child must exist")
    parent_time = int(nodes[parent_id][0])
    if int(nodes[existing_child_id][0]) != parent_time + 1:
        raise ValueError("the existing child must be in the next frame")
    has_parent = {target for _source, target in edges}
    parent_position = np.asarray(nodes[parent_id][1:], dtype=np.float64) * VOXEL_SCALE_UM
    child_position = (
        np.asarray(nodes[existing_child_id][1:], dtype=np.float64) * VOXEL_SCALE_UM
    )
    candidates: list[dict[str, Any]] = []
    for node_id, row in nodes.items():
        if int(row[0]) != parent_time + 1 or node_id == existing_child_id:
            continue
        position = np.asarray(row[1:], dtype=np.float64) * VOXEL_SCALE_UM
        candidates.append(
            {
                "node_id": int(node_id),
                "parent_distance_um": float(np.linalg.norm(position - parent_position)),
                "sister_distance_um": float(np.linalg.norm(position - child_position)),
                "parent_free": node_id not in has_parent,
            }
        )
    return sorted(
        candidates,
        key=lambda row: (
            not bool(row["parent_free"]),
            float(row["parent_distance_um"]),
            float(row["sister_distance_um"]),
            int(row["node_id"]),
        ),
    )


def audit_movie(prediction_path: Path, truth_path: Path) -> dict[str, Any]:
    pred_nodes, pred_edges = graph_plain(prediction_path)
    truth_nodes, truth_edges = graph_plain(truth_path)
    pred_to_truth, truth_to_pred = match_nodes_bipartite(
        pred_nodes, truth_nodes, max_dist=7.0
    )
    _pred_to_truth_3, truth_to_pred_3 = match_nodes_bipartite(
        pred_nodes, truth_nodes, max_dist=3.0
    )
    truth_out: dict[int, set[int]] = {}
    pred_out: dict[int, set[int]] = {}
    for source, target in truth_edges:
        truth_out.setdefault(source, set()).add(target)
    for source, target in pred_edges:
        pred_out.setdefault(source, set()).add(target)
    pred_edge_set = set(pred_edges)
    oracle_edges = list(pred_edges)
    events: list[dict[str, Any]] = []
    for truth_parent, truth_children_set in sorted(truth_out.items()):
        if len(truth_children_set) < 2:
            continue
        truth_children = sorted(truth_children_set)[:2]
        truth_event = [truth_parent, *truth_children]
        mapped = [truth_to_pred.get(node_id) for node_id in truth_event]
        mapped_3um = [truth_to_pred_3.get(node_id) for node_id in truth_event]
        detected_triplet = all(node_id is not None for node_id in mapped)
        direct_edges = [
            mapped[0] is not None
            and child is not None
            and (int(mapped[0]), int(child)) in pred_edge_set
            for child in mapped[1:]
        ]
        event: dict[str, Any] = {
            "truth_parent_id": int(truth_parent),
            "timepoint": int(truth_nodes[truth_parent][0]),
            "mapped_prediction_ids_7um": mapped,
            "mapped_prediction_ids_3um": mapped_3um,
            "detected_triplet_7um": detected_triplet,
            "direct_daughter_edges": direct_edges,
        }
        if detected_triplet:
            typed_mapped = [int(node_id) for node_id in mapped if node_id is not None]
            oracle_edges.extend(
                (typed_mapped[0], child_id) for child_id in typed_mapped[1:]
            )
            if sum(direct_edges) == 1:
                existing_child = typed_mapped[1 + direct_edges.index(True)]
                missing_child = typed_mapped[1 + direct_edges.index(False)]
                candidates = ranked_second_daughter_candidates(
                    pred_nodes,
                    pred_edges,
                    parent_id=typed_mapped[0],
                    existing_child_id=existing_child,
                )
                missing_row = next(
                    row for row in candidates if row["node_id"] == missing_child
                )
                parent_free = [row for row in candidates if row["parent_free"]]
                by_parent_distance = sorted(
                    candidates,
                    key=lambda row: (
                        float(row["parent_distance_um"]),
                        float(row["sister_distance_um"]),
                        int(row["node_id"]),
                    ),
                )
                by_sister_distance = sorted(
                    candidates,
                    key=lambda row: (
                        float(row["sister_distance_um"]),
                        float(row["parent_distance_um"]),
                        int(row["node_id"]),
                    ),
                )
                event["recoverable_single_missing_daughter"] = True
                event["missing_daughter"] = {
                    **missing_row,
                    "rank_by_parent_distance_all": 1
                    + next(
                        index
                        for index, row in enumerate(by_parent_distance)
                        if row["node_id"] == missing_child
                    ),
                    "rank_by_sister_distance_all": 1
                    + next(
                        index
                        for index, row in enumerate(by_sister_distance)
                        if row["node_id"] == missing_child
                    ),
                    "rank_among_parent_free": (
                        1
                        + next(
                            index
                            for index, row in enumerate(parent_free)
                            if row["node_id"] == missing_child
                        )
                        if bool(missing_row["parent_free"])
                        else None
                    ),
                    "requires_parent_reassignment": not bool(
                        missing_row["parent_free"]
                    ),
                    "parent_free_candidates": len(parent_free),
                }
            else:
                event["recoverable_single_missing_daughter"] = False
        events.append(event)
    base_division = compute_division_confusion(
        pred_nodes,
        pred_edges,
        truth_nodes,
        truth_edges,
        pred_to_truth,
        truth_to_pred,
    )
    oracle_edges = sorted(set(oracle_edges))
    oracle_division = compute_division_confusion(
        pred_nodes,
        oracle_edges,
        truth_nodes,
        truth_edges,
        pred_to_truth,
        truth_to_pred,
    )
    base_edge = compute_edge_confusion(pred_edges, truth_edges, pred_to_truth)
    oracle_edge = compute_edge_confusion(oracle_edges, truth_edges, pred_to_truth)
    return {
        "stem": truth_path.stem,
        "predicted_nodes": len(pred_nodes),
        "predicted_edges": len(pred_edges),
        "division_events": events,
        "base_edge_tp_fp_fn": list(base_edge),
        "oracle_detected_triplet_edge_tp_fp_fn": list(oracle_edge),
        "base_division_tp_fp_fn": list(base_division),
        "oracle_detected_triplet_division_tp_fp_fn": list(oracle_division),
    }


def find_unique_graph(root: Path, stem: str) -> Path:
    matches = sorted(path for path in root.rglob(f"{stem}.geff") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"expected one prediction graph for {stem}, saw {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    truth_paths = sorted(args.truth_root.glob("*.geff"))
    if not truth_paths:
        raise FileNotFoundError("no truth GEFFs found")
    movies = [
        audit_movie(
            find_unique_graph(args.prediction_root, truth_path.stem), truth_path
        )
        for truth_path in truth_paths
    ]
    events = [event for movie in movies for event in movie["division_events"]]
    result = {
        "schema_version": 1,
        "status": "oracle_feasibility_only",
        "movies": movies,
        "summary": {
            "movies": len(movies),
            "truth_division_events": len(events),
            "detected_triplets_7um": sum(
                bool(event["detected_triplet_7um"]) for event in events
            ),
            "single_missing_daughter_edges": sum(
                bool(event.get("recoverable_single_missing_daughter"))
                for event in events
            ),
            "threshold_selection_authorized": False,
            "submission_authorized": False,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
