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
    from trainer import (
        GraphVideo,
        hybrid_link_movie,
        link_movie,
        predict_movie_scores,
        read_graph_video,
    )
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
        read_graph_video,
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


def load_acceptance_evidence(
    terminal_path: Path, model_path: Path
) -> dict[str, Any]:
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "completed":
        raise RuntimeError("Association acceptance terminal is not complete")
    if not terminal.get("association_acceptance_passed"):
        raise RuntimeError("Association model failed frozen clean acceptance")
    model_sha256 = sha256_file(model_path)
    if model_sha256 != terminal.get("model_sha256"):
        raise RuntimeError("Association checkpoint does not match acceptance evidence")
    if "complete_movie_selected" not in terminal:
        raise RuntimeError("Acceptance evidence has no frozen selected configuration")
    return terminal


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


def transfer_raw_edge_probabilities(
    videos: dict[str, GraphVideo],
    raw_videos: dict[str, GraphVideo],
    *,
    fallback_probability: float = 0.80,
    minimum_node_id_coverage: float = 0.90,
) -> dict[str, dict[str, float | int]]:
    """Transfer pre-postprocessing edge confidence onto final CSV edges by ID."""
    if set(videos) != set(raw_videos):
        raise ValueError("Raw graph datasets do not match submission datasets")
    report: dict[str, dict[str, float | int]] = {}
    for stem, video in sorted(videos.items()):
        raw = raw_videos[stem]
        shared_nodes = set(map(int, video.node_ids)) & set(map(int, raw.node_ids))
        node_coverage = len(shared_nodes) / max(len(video.node_ids), 1)
        if node_coverage < minimum_node_id_coverage:
            raise ValueError(
                f"{stem}: raw graph node-ID coverage {node_coverage:.4f} is below "
                f"{minimum_node_id_coverage:.4f}"
            )
        raw_probability = {
            (int(source), int(target)): float(probability)
            for (source, target), probability in zip(
                raw.edges.tolist(), raw.edge_probabilities.tolist()
            )
            if np.isfinite(probability)
        }
        probabilities = np.asarray(
            [
                raw_probability.get(
                    (int(source), int(target)), float(fallback_probability)
                )
                for source, target in video.edges.tolist()
            ],
            dtype=np.float32,
        )
        transferred = sum(
            (int(source), int(target)) in raw_probability
            for source, target in video.edges.tolist()
        )
        video.edge_probabilities = probabilities
        report[stem] = {
            "nodes": len(video.node_ids),
            "shared_node_ids": len(shared_nodes),
            "node_id_coverage": node_coverage,
            "edges": len(video.edges),
            "transferred_edge_probabilities": transferred,
            "edge_probability_coverage": transferred / max(len(video.edges), 1),
            "fallback_edges": len(video.edges) - transferred,
            "fallback_probability": float(fallback_probability),
        }
    return report


def read_raw_graphs(root: Path, stems: set[str]) -> dict[str, GraphVideo]:
    """Resolve exactly one complete raw GEFF per requested test stem."""
    result: dict[str, GraphVideo] = {}
    for stem in sorted(stems):
        candidates = [
            path
            for path in root.rglob(f"{stem}.geff")
            if (path / "zarr.json").is_file()
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one raw graph for {stem} below {root}, found {candidates}"
            )
        result[stem] = read_graph_video(candidates[0])
    return result


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
    if selected["method"] not in {
        "submission_graph_hybrid",
        "raw_confidence_hybrid",
    }:
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
        use_stored_edge_probabilities=selected["method"] == "raw_confidence_hybrid",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--acceptance-terminal", type=Path)
    parser.add_argument("--trackastra-dir", type=Path, required=True)
    parser.add_argument("--base-graph-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--candidate-radius", type=float, default=80.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "model.pt"
    terminal_path = args.acceptance_terminal or (
        args.model_dir / "training_terminal.json"
    )
    terminal = load_acceptance_evidence(terminal_path, model_path)
    model_sha256 = sha256_file(model_path)
    selected = terminal["complete_movie_selected"]

    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for full submission association inference")
    model = TrackingTransformer.from_folder(args.model_dir, map_location="cpu").to(device)
    model.eval()
    videos = read_submission(args.base_submission)
    edge_probability_transfer = None
    if args.base_graph_root is not None:
        raw_videos = read_raw_graphs(args.base_graph_root, set(videos))
        edge_probability_transfer = transfer_raw_edge_probabilities(videos, raw_videos)
    if selected["method"] == "raw_confidence_hybrid" and edge_probability_transfer is None:
        raise RuntimeError("Raw-confidence hybrid requires --base-graph-root")
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
        "acceptance_evidence_sha256": sha256_file(terminal_path),
        "acceptance_evaluation_kind": terminal.get(
            "evaluation_kind", "training_terminal"
        ),
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
        "edge_probability_transfer": edge_probability_transfer,
        "total_changed_edges": changed_edges,
        "nodes_preserved_exactly": True,
        "public_leaderboard_used_for_selection": False,
    }
    atomic_json(args.output_dir / "candidate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
