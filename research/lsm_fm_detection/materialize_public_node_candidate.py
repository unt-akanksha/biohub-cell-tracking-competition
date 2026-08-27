#!/usr/bin/env python
"""Run the frozen public postprocessor with LSM-FM node coordinates.

The public raw graph supplies node identities, frame assignments, edge
topology, and edge confidence. A topology-identical candidate supplies only
the replacement z/y/x coordinates. The public intensity-centroid pass is
therefore skipped; every later frozen postprocessing stage is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from materialize_public_validation import (
        EXPECTED_DEEPCENTER_SHA256,
        EXPECTED_PROCESSED_NODES,
        EXPECTED_STEMS,
        FIELDNAMES,
        atomic_json,
        load_public_namespace,
        sha256_file,
        validate_processed_node_count,
    )
except ModuleNotFoundError:
    from research.trackastra_graph.materialize_public_validation import (
        EXPECTED_DEEPCENTER_SHA256,
        EXPECTED_PROCESSED_NODES,
        EXPECTED_STEMS,
        FIELDNAMES,
        atomic_json,
        load_public_namespace,
        sha256_file,
        validate_processed_node_count,
    )


def assert_raw_topology_compatible(
    control_nodes: Mapping[int, int],
    control_edges: Iterable[tuple[int, int]],
    candidate_nodes: Mapping[int, int],
    candidate_edges: Iterable[tuple[int, int]],
    *,
    stem: str,
) -> None:
    """Require exact node-ID/frame and edge equality before coordinate use."""

    if dict(control_nodes) != dict(candidate_nodes):
        raise RuntimeError(f"{stem}: candidate changed node IDs or frame assignments")
    if sorted(control_edges) != sorted(candidate_edges):
        raise RuntimeError(f"{stem}: candidate changed public raw topology")


def graph_records(graph: Any) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[int, dict[str, Any]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = {
            "node_id": node_id,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
    edges: list[dict[str, Any]] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        probability = row.get("edge_prob") if hasattr(row, "get") else None
        edges.append(
            {
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if probability is None else float(probability),
            }
        )
    return nodes, edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-raw-root", type=Path, required=True)
    parser.add_argument("--candidate-raw-root", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--public-preset-source", type=Path, required=True)
    parser.add_argument("--public-config-source", type=Path, required=True)
    parser.add_argument("--public-postprocess-source", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
    if not (
        source_result.get("status") == "completed"
        and source_result.get("promotion_passed") is True
        and source_result.get("selection_mode") == "predeclared_without_label_access"
        and source_result.get("hyperparameter_selection_performed") is False
        and source_result.get("public_graph_used_as_base") is True
        and source_result.get("exact_public_replica") is False
        and source_result.get("competition_submission_performed") is False
    ):
        raise ValueError("public-node source result is not production-gate eligible")

    controls = {
        path.stem: path
        for path in args.control_raw_root.glob("*.geff")
        if (path / "zarr.json").is_file()
    }
    candidates = {
        path.stem: path
        for path in args.candidate_raw_root.glob("*.geff")
        if (path / "zarr.json").is_file()
    }
    if set(controls) != EXPECTED_STEMS or set(candidates) != EXPECTED_STEMS:
        raise RuntimeError(
            "Four-movie graph coverage mismatch: "
            f"control={sorted(controls)}, candidate={sorted(candidates)}"
        )

    namespace = load_public_namespace(
        args.public_preset_source,
        args.public_config_source,
        args.public_postprocess_source,
        args.competition_dir,
    )
    graph_from_geff = namespace["graph_from_geff"]
    filter_output_graph = namespace["filter_output_graph"]
    deepcenter = namespace["load_deepcenter_veto_detector"]()
    if deepcenter is None:
        raise RuntimeError("Hash-pinned public DeepCenter checkpoint was not loaded")
    deepcenter_path = Path(deepcenter["path"])
    if sha256_file(deepcenter_path) != EXPECTED_DEEPCENTER_SHA256:
        raise RuntimeError("DeepCenter checkpoint hash mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "processed_candidate.csv"
    temporary = csv_path.with_suffix(".tmp")
    stats: dict[str, dict[str, Any]] = {}
    row_id = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for stem in sorted(EXPECTED_STEMS):
            control_graph = graph_from_geff(controls[stem])
            candidate_graph = graph_from_geff(candidates[stem])
            control_nodes, control_edges = graph_records(control_graph)
            candidate_nodes, candidate_edges = graph_records(candidate_graph)
            assert_raw_topology_compatible(
                {node_id: row["t"] for node_id, row in control_nodes.items()},
                ((row["source_id"], row["target_id"]) for row in control_edges),
                {node_id: row["t"] for node_id, row in candidate_nodes.items()},
                ((row["source_id"], row["target_id"]) for row in candidate_edges),
                stem=stem,
            )
            processed_nodes, processed_edges, filter_stats = filter_output_graph(
                candidate_nodes,
                control_edges,
                dataset=stem,
                deepcenter_bundle=deepcenter,
            )
            if not processed_nodes:
                raise RuntimeError(f"{stem}: public postprocessor removed every node")
            tolerance = validate_processed_node_count(
                stem, len(processed_nodes), EXPECTED_PROCESSED_NODES[stem]
            )
            for node_id in sorted(processed_nodes):
                node = processed_nodes[node_id]
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": stem,
                        "row_type": "node",
                        "node_id": int(node_id),
                        "t": int(node["t"]),
                        "z": max(0, int(round(float(node["z"])))),
                        "y": max(0, int(round(float(node["y"])))),
                        "x": max(0, int(round(float(node["x"])))),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1
            for edge in processed_edges:
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": stem,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": int(edge["source_id"]),
                        "target_id": int(edge["target_id"]),
                    }
                )
                row_id += 1
            stats[stem] = {
                "raw_nodes": len(candidate_nodes),
                "raw_edges": len(control_edges),
                "processed_nodes": len(processed_nodes),
                "processed_edges": len(processed_edges),
                "expected_processed_nodes": EXPECTED_PROCESSED_NODES[stem],
                "processed_node_tolerance": tolerance,
                "postprocess": filter_stats,
            }
    temporary.replace(csv_path)
    report = {
        "schema_version": 1,
        "status": "completed",
        "role": "production-aligned public topology with independent LSM-FM coordinates",
        "source_result_sha256": sha256_file(args.source_result),
        "processed_candidate_sha256": sha256_file(csv_path),
        "datasets": stats,
        "public_graph_used_as_base": True,
        "public_predictions_copied": True,
        "exact_public_replica": False,
        "coordinate_source": "independent_lsm_fm_probability",
        "public_intensity_centroid_replaced": True,
        "ground_truth_read_for_postprocessing": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
    }
    atomic_json(args.output_dir / "processed_candidate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
