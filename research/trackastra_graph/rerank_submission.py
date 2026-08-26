from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from hybrid_linker import HybridLinkConfig
    from trainer import GraphVideo, hybrid_link_movie, link_movie, predict_movie_scores
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from research.trackastra_graph.hybrid_linker import HybridLinkConfig
    from research.trackastra_graph.train_biohub_graph_transformer import (
        GraphVideo,
        hybrid_link_movie,
        link_movie,
        predict_movie_scores,
    )


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_submission(path: Path) -> dict[str, GraphVideo]:
    nodes: dict[str, list[tuple[int, int, float, float, float]]] = {}
    edges: dict[str, list[tuple[int, int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDNAMES:
            raise ValueError(
                f"Unexpected submission columns: {reader.fieldnames}; expected {FIELDNAMES}"
            )
        for row_number, row in enumerate(reader, start=2):
            stem = row["dataset"]
            if not stem:
                raise ValueError(f"Row {row_number}: empty dataset")
            if row["row_type"] == "node":
                nodes.setdefault(stem, []).append(
                    (
                        int(row["node_id"]),
                        int(row["t"]),
                        float(row["z"]),
                        float(row["y"]),
                        float(row["x"]),
                    )
                )
            elif row["row_type"] == "edge":
                edges.setdefault(stem, []).append(
                    (int(row["source_id"]), int(row["target_id"]))
                )
            else:
                raise ValueError(f"Row {row_number}: invalid row_type={row['row_type']!r}")

    if not nodes:
        raise ValueError("Base submission contains no node rows")
    videos: dict[str, GraphVideo] = {}
    for stem in sorted(nodes):
        rows = nodes[stem]
        videos[stem] = GraphVideo(
            stem=stem,
            node_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
            times=np.asarray([row[1] for row in rows], dtype=np.int32),
            coords_voxel=np.asarray([row[2:] for row in rows], dtype=np.float32),
            edges=np.asarray(edges.get(stem, ()), dtype=np.int64).reshape(-1, 2),
        )
    extra_edge_datasets = set(edges) - set(videos)
    if extra_edge_datasets:
        raise ValueError(f"Edges without node datasets: {sorted(extra_edge_datasets)}")
    return videos


def validate_edges(video: GraphVideo, edges: list[tuple[int, int]]) -> None:
    time_by_id = video.time_by_id
    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    for source, target in edges:
        if source not in time_by_id or target not in time_by_id:
            raise ValueError(f"{video.stem}: edge references missing node {source}->{target}")
        if time_by_id[target] != time_by_id[source] + 1:
            raise ValueError(f"{video.stem}: nonconsecutive edge {source}->{target}")
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1
    if incoming and max(incoming.values()) > 1:
        raise ValueError(f"{video.stem}: candidate has a node with multiple parents")
    if outgoing and max(outgoing.values()) > 2:
        raise ValueError(f"{video.stem}: candidate has a node with more than two children")


def write_submission(
    path: Path,
    videos: dict[str, GraphVideo],
    candidate_edges: dict[str, list[tuple[int, int]]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    row_id = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for stem, video in sorted(videos.items()):
            node_rows = sorted(
                zip(
                    video.node_ids.tolist(),
                    video.times.tolist(),
                    video.coords_voxel.tolist(),
                ),
                key=lambda item: (int(item[1]), int(item[0])),
            )
            for node_id, time, coords in node_rows:
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": stem,
                        "row_type": "node",
                        "node_id": int(node_id),
                        "t": int(time),
                        "z": float(coords[0]),
                        "y": float(coords[1]),
                        "x": float(coords[2]),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1
            for source, target in sorted(candidate_edges[stem]):
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
                        "source_id": int(source),
                        "target_id": int(target),
                    }
                )
                row_id += 1
    temporary.replace(path)


def selected_edges_for_video(
    model,
    video: GraphVideo,
    selected: dict[str, Any],
    device: torch.device,
    *,
    max_tokens: int,
    candidate_radius: float,
) -> list[tuple[int, int]]:
    pair_scores = predict_movie_scores(
        model,
        video,
        device,
        max_tokens=max_tokens,
        candidate_radius=candidate_radius,
    )
    if selected["method"] == "trackastra_only":
        return link_movie(
            pair_scores,
            edge_threshold=float(selected["edge_threshold"]),
            division_threshold=float(selected["division_threshold"]),
            division_ratio=float(selected["division_ratio"]),
        )
    if selected["method"] != "submission_graph_hybrid":
        raise ValueError(f"Unknown selected association method: {selected['method']}")
    config = HybridLinkConfig(
        edge_threshold=float(selected["edge_threshold"]),
        base_lock_probability=float(selected["base_lock_probability"]),
        base_keep_probability=float(selected["base_keep_probability"]),
        base_bonus=float(selected["base_bonus"]),
        division_threshold=float(selected["division_threshold"]),
        division_ratio=float(selected["division_ratio"]),
        base_division_keep_probability=float(selected["base_division_keep_probability"]),
    )
    return hybrid_link_movie(
        video,
        pair_scores,
        config,
        submission_edge_probability=float(selected["base_pseudo_probability"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trackastra-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--candidate-radius", type=float, default=80.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = args.model_dir / "training_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "completed":
        raise RuntimeError("Association training terminal is not complete")
    if not terminal.get("association_acceptance_passed"):
        raise RuntimeError("Association model failed frozen clean acceptance")
    model_path = args.model_dir / "model.pt"
    model_sha256 = sha256_file(model_path)
    if model_sha256 != terminal.get("model_sha256"):
        raise RuntimeError("Association checkpoint does not match training terminal")
    selected = terminal["complete_movie_selected"]

    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for full submission association inference")
    model = TrackingTransformer.from_folder(args.model_dir, map_location="cpu").to(device)
    model.eval()
    videos = read_submission(args.base_submission)
    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, int]] = {}
    for stem, video in sorted(videos.items()):
        print(
            f"ASSOCIATION INFERENCE {stem}: nodes={len(video.node_ids)} "
            f"base_edges={len(video.edges)}",
            flush=True,
        )
        edges = selected_edges_for_video(
            model,
            video,
            selected,
            device,
            max_tokens=args.max_tokens,
            candidate_radius=args.candidate_radius,
        )
        validate_edges(video, edges)
        candidate_edges[stem] = edges
        base_set = set(map(tuple, video.edges.tolist()))
        candidate_set = set(edges)
        dataset_stats[stem] = {
            "nodes": len(video.node_ids),
            "base_edges": len(base_set),
            "candidate_edges": len(candidate_set),
            "retained_edges": len(base_set & candidate_set),
            "removed_edges": len(base_set - candidate_set),
            "new_edges": len(candidate_set - base_set),
        }

    output_path = args.output_dir / "submission.csv"
    write_submission(output_path, videos, candidate_edges)
    changed_edges = sum(
        item["removed_edges"] + item["new_edges"] for item in dataset_stats.values()
    )
    if changed_edges <= 0:
        raise RuntimeError("Learned association produced an exact base-edge replica")
    if sha256_file(output_path) == sha256_file(args.base_submission):
        raise RuntimeError("Candidate submission is byte-identical to the public base")

    report = {
        "schema_version": 1,
        "status": "completed",
        "base_submission_sha256": sha256_file(args.base_submission),
        "candidate_submission_sha256": sha256_file(output_path),
        "model_sha256": model_sha256,
        "association_method": selected["method"],
        "association_configuration": {
            key: value
            for key, value in selected.items()
            if key not in {"selection_summary", "acceptance_summary", "all_summary"}
        },
        "clean_selection_summary": selected["selection_summary"],
        "clean_acceptance_summary": selected["acceptance_summary"],
        "clean_acceptance_delta_vs_base_raw": terminal[
            "acceptance_delta_vs_base_raw"
        ],
        "datasets": dataset_stats,
        "total_changed_edges": changed_edges,
        "nodes_preserved_exactly": True,
        "public_leaderboard_used_for_selection": False,
    }
    atomic_json(args.output_dir / "candidate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
