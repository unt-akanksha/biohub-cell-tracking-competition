from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDNAMES = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
EXPECTED_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}
EXPECTED_PROCESSED_NODES = {
    "44b6_12dfb391": 44139,
    "44b6_267148e4": 21768,
    "6bba_062c8d37": 5812,
    "6bba_07e24132": 26204,
}
EXPECTED_DEEPCENTER_SHA256 = (
    "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_public_namespace(
    preset_source: Path,
    config_source: Path,
    postprocess_source: Path,
    competition_dir: Path,
) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "public_validation_postprocessor"}
    exec(
        compile(preset_source.read_text(encoding="utf-8"), str(preset_source), "exec"),
        namespace,
    )
    exec(
        compile(config_source.read_text(encoding="utf-8"), str(config_source), "exec"),
        namespace,
    )
    namespace["COMP_DIR"] = competition_dir
    namespace["TEST_DIR"] = competition_dir / "train"
    namespace["WORKING_DIR"] = Path("/kaggle/working")
    exec(
        compile(
            postprocess_source.read_text(encoding="utf-8"),
            str(postprocess_source),
            "exec",
        ),
        namespace,
    )
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-validation-root", type=Path, required=True)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--public-preset-source", type=Path, required=True)
    parser.add_argument("--public-config-source", type=Path, required=True)
    parser.add_argument("--public-postprocess-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    geffs = {
        path.stem: path
        for path in args.raw_validation_root.glob("*.geff")
        if (path / "zarr.json").is_file()
    }
    if set(geffs) != EXPECTED_STEMS:
        raise RuntimeError(
            f"Frozen validation graph mismatch: expected={sorted(EXPECTED_STEMS)}, "
            f"found={sorted(geffs)}"
        )
    namespace = load_public_namespace(
        args.public_preset_source,
        args.public_config_source,
        args.public_postprocess_source,
        args.competition_dir,
    )
    graph_from_geff = namespace["graph_from_geff"]
    refine_all_centroids = namespace["refine_all_centroids"]
    filter_output_graph = namespace["filter_output_graph"]
    deepcenter = namespace["load_deepcenter_veto_detector"]()
    if deepcenter is None:
        raise RuntimeError("Hash-pinned public DeepCenter checkpoint was not loaded")
    deepcenter_path = Path(deepcenter["path"])
    deepcenter_sha256 = sha256_file(deepcenter_path)
    if deepcenter_sha256 != EXPECTED_DEEPCENTER_SHA256:
        raise RuntimeError(
            "DeepCenter checkpoint hash mismatch: "
            f"expected {EXPECTED_DEEPCENTER_SHA256}, got {deepcenter_sha256} "
            f"from {deepcenter_path}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "processed_validation.csv"
    temporary = csv_path.with_suffix(".tmp")
    stats: dict[str, dict[str, Any]] = {}
    row_id = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for stem, path in sorted(geffs.items()):
            graph = graph_from_geff(path)
            nodes_by_id = {
                int(row["node_id"]): {
                    "node_id": int(row["node_id"]),
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }
                for row in graph.node_attrs().iter_rows(named=True)
            }
            raw_edges = []
            for row in graph.edge_attrs().iter_rows(named=True):
                probability = row.get("edge_prob") if hasattr(row, "get") else None
                raw_edges.append(
                    {
                        "source_id": int(row["source_id"]),
                        "target_id": int(row["target_id"]),
                        "edge_prob": (
                            None if probability is None else float(probability)
                        ),
                    }
                )
            raw_nodes = len(nodes_by_id)
            nodes_by_id = refine_all_centroids(nodes_by_id, stem)
            nodes_by_id, edges, filter_stats = filter_output_graph(
                nodes_by_id,
                raw_edges,
                dataset=stem,
                deepcenter_bundle=deepcenter,
            )
            if not nodes_by_id:
                raise RuntimeError(f"{stem}: public postprocessor removed every node")
            expected_nodes = EXPECTED_PROCESSED_NODES[stem]
            if len(nodes_by_id) != expected_nodes:
                raise RuntimeError(
                    f"{stem}: processed node count {len(nodes_by_id)} does not match "
                    f"the hash-pinned public validator result {expected_nodes}"
                )

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
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
            for edge in edges:
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
                "raw_nodes": raw_nodes,
                "processed_nodes": len(nodes_by_id),
                "expected_processed_nodes": expected_nodes,
                "raw_edges": len(raw_edges),
                "processed_edges": len(edges),
                "postprocess": filter_stats,
            }
    temporary.replace(csv_path)
    report = {
        "schema_version": 1,
        "status": "completed",
        "source_notebook": "evgendvorkin/biohub-0-927-lb",
        "role": "frozen comparator postprocessing only",
        "public_preset_source_sha256": sha256_file(args.public_preset_source),
        "public_config_source_sha256": sha256_file(args.public_config_source),
        "public_postprocess_source_sha256": sha256_file(
            args.public_postprocess_source
        ),
        "processed_validation_sha256": sha256_file(csv_path),
        "deepcenter_checkpoint": {
            "path": str(deepcenter_path),
            "sha256": deepcenter_sha256,
            "expected_epoch": int(namespace["DEEPCENTER_EXPECTED_EPOCH"]),
        },
        "datasets": stats,
        "ground_truth_read_for_postprocessing": False,
        "public_leaderboard_used_for_selection": False,
    }
    atomic_json(args.output_dir / "processed_validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
