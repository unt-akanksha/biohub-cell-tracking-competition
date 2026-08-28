#!/usr/bin/env python
"""Measure a division-only coordinate intervention without reading truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from biohub_tracker.graphs import (
    GraphData,
    GraphNode,
    artifact_tree_sha256,
    graph_data_from_tracksdata,
    load_geff_graph,
)
from biohub_tracker.io import atomic_write_json, sha256_file
from biohub_tracker.manifests import load_manifest
from biohub_tracker.scorer_lock import verify_scorer_lock
from research.division_localization_refinement import (
    apply_division_coordinate_donor,
    division_coordinate_donor_scope,
    division_localization_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--processed-control-csv", type=Path, required=True)
    parser.add_argument("--refined-dir", type=Path, required=True)
    parser.add_argument("--scorer-lock", type=Path, required=True)
    parser.add_argument("--organizer-checkout", type=Path, required=True)
    parser.add_argument("--tracksdata-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def graph_data_from_submission(table: pl.DataFrame, stem: str) -> GraphData:
    movie = table.filter(pl.col("dataset") == stem)
    node_rows = movie.filter(pl.col("row_type") == "node")
    edge_rows = movie.filter(pl.col("row_type") == "edge")
    if not node_rows.height:
        raise ValueError(f"processed control omits node rows for {stem}")
    return GraphData(
        nodes=tuple(
            GraphNode(row["node_id"], row["t"], row["z"], row["y"], row["x"])
            for row in node_rows.select("node_id", "t", "z", "y", "x").iter_rows(named=True)
        ),
        edges=tuple(
            (row["source_id"], row["target_id"])
            for row in edge_rows.select("source_id", "target_id").iter_rows(named=True)
        ),
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite scope evidence: {args.output}")
    manifest = load_manifest(args.manifest)
    samples = {sample.sample_id: sample for sample in manifest.samples}
    verified = verify_scorer_lock(
        args.scorer_lock,
        args.organizer_checkout,
        tracksdata_checkout=args.tracksdata_checkout,
    )
    control_table = pl.read_csv(args.processed_control_csv)
    required_columns = {
        "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
    }
    if not required_columns <= set(control_table.columns):
        raise ValueError("processed control CSV has an invalid schema")
    stems = sorted(str(value) for value in control_table["dataset"].unique())
    refined_stems = sorted(path.stem for path in args.refined_dir.glob("*.geff") if path.is_dir())
    if not stems or stems != refined_stems:
        raise ValueError("processed control and donor inventories must be identical and nonempty")
    if any(stem not in samples for stem in stems):
        raise ValueError("manifest omits a prediction movie")

    rows: list[dict[str, Any]] = []
    all_displacements: list[float] = []
    for stem in stems:
        refined_path = args.refined_dir / f"{stem}.geff"
        control = graph_data_from_submission(control_table, stem)
        refined = graph_data_from_tracksdata(load_geff_graph(refined_path, verified))
        output = apply_division_coordinate_donor(control, refined)
        scope = division_localization_scope(control)
        donor_scope = division_coordinate_donor_scope(control, refined)
        control_nodes = {int(node.node_id): node for node in control.nodes}
        output_nodes = {int(node.node_id): node for node in output.nodes}
        scale = tuple(float(value) for value in samples[stem].scale_zyx_um)
        displacements: list[float] = []
        changed = 0
        for node_id in donor_scope.eligible_node_ids:
            before, after = control_nodes[node_id], output_nodes[node_id]
            delta = tuple(
                (float(right) - float(left)) * factor
                for left, right, factor in zip(
                    (before.z, before.y, before.x),
                    (after.z, after.y, after.x),
                    scale,
                    strict=True,
                )
            )
            distance = math.sqrt(sum(value * value for value in delta))
            displacements.append(distance)
            changed += int(distance > 0.0)
        all_displacements.extend(displacements)
        rows.append(
            {
                "sample_id": stem,
                "node_count": len(control.nodes),
                "edge_count": len(control.edges),
                "division_parent_count": len(scope.division_parent_ids),
                "division_daughter_count": len(scope.division_daughter_ids),
                "eligible_division_parent_count": len(
                    donor_scope.eligible_division_parent_ids
                ),
                "unsupported_division_parent_count": len(
                    donor_scope.unsupported_division_parent_ids
                ),
                "missing_donor_node_count": len(donor_scope.missing_donor_node_ids),
                "selected_node_count": len(donor_scope.eligible_node_ids),
                "selected_node_fraction": len(donor_scope.eligible_node_ids) / len(control.nodes),
                "changed_selected_node_count": changed,
                "max_selected_displacement_um": max(displacements, default=0.0),
                "median_selected_displacement_um": quantile(displacements, 0.5),
                "refined_graph_sha256": artifact_tree_sha256(refined_path),
                "topology_and_time_preserved": True,
            }
        )

    total_nodes = sum(row["node_count"] for row in rows)
    total_selected = sum(row["selected_node_count"] for row in rows)
    result = {
        "schema_version": 1,
        "profile_id": "division-localization-scope-v1",
        "status": "completed",
        "evaluation_kind": "label_free_mechanical_scope",
        "ground_truth_read": False,
        "official_scoring_performed": False,
        "hyperparameter_selection_performed": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "coordinate_scope": "predicted division parents and their immediate daughters",
        "graph_inventory_count": len(rows),
        "total_node_count": total_nodes,
        "total_selected_node_count": total_selected,
        "total_selected_node_fraction": total_selected / total_nodes,
        "changed_selected_node_count": sum(row["changed_selected_node_count"] for row in rows),
        "selected_displacement_um": {
            "median": quantile(all_displacements, 0.5),
            "p90": quantile(all_displacements, 0.9),
            "maximum": max(all_displacements, default=0.0),
        },
        "movies": rows,
        "manifest_sha256": sha256_file(args.manifest),
        "processed_control_csv_sha256": sha256_file(args.processed_control_csv),
        "implementation_sha256": sha256_file(Path(__file__).parents[1] / "research/division_localization_refinement.py"),
        "scorer_lock_sha256": verified.lock_sha256,
        "note": "Prediction-only scope audit. Opened truth artifacts were neither addressed nor loaded.",
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
