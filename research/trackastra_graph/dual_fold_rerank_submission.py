"""Two-GPU whole-movie inference for an accepted reciprocal Trackastra pair.

This module deliberately cannot tune an association configuration.  It consumes
one hash-bound acceptance document containing an already frozen configuration,
routes each known embryo prefix to the model trained without that embryo, and
balances complete movies between exactly two isolated CUDA workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import rerank_submission as rerank
except ModuleNotFoundError:
    from research.trackastra_graph import rerank_submission as rerank

try:
    from submission_sharding import (
        build_movie_shards,
        shard_plan_sha256,
        validate_shard_outputs,
        visible_cuda_tokens,
        worker_environment,
    )
except ModuleNotFoundError:
    from research.submission_sharding import (
        build_movie_shards,
        shard_plan_sha256,
        validate_shard_outputs,
        visible_cuda_tokens,
        worker_environment,
    )


MODEL_KEYS = {"44b6": "target_44b6", "6bba": "target_6bba"}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embryo_prefix(stem: str) -> str:
    prefix = str(stem).split("_", maxsplit=1)[0]
    if prefix not in MODEL_KEYS:
        raise ValueError(f"no reciprocal model is declared for dataset prefix {prefix}")
    return prefix


def movie_inference_weight(video: rerank.GraphVideo) -> float:
    """Estimate pair-scoring work without cutting a movie into frame shards."""

    pair_products = 0
    for timepoint, source_ids in video.ids_by_time.items():
        target_ids = video.ids_by_time.get(timepoint + 1)
        if target_ids is not None:
            pair_products += len(source_ids) * len(target_ids)
    return float(max(pair_products, len(video.node_ids), 1))


def load_acceptance(path: Path, model_dirs: dict[str, Path]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "accepted":
        raise RuntimeError("dual-fold evidence is not accepted")
    if payload.get("evaluation_kind") != "exact_processed_dual_fold_acceptance":
        raise RuntimeError("unexpected dual-fold acceptance kind")
    if payload.get("public_leaderboard_used_for_selection") is not False:
        raise RuntimeError("leaderboard-selected evidence is forbidden")
    if payload.get("exact_processed_gate_passed") is not True:
        raise RuntimeError("exact processed comparator gate did not pass")
    if not isinstance(payload.get("association_configuration"), dict):
        raise RuntimeError("accepted association configuration is missing")
    expected_models = payload.get("models")
    if not isinstance(expected_models, dict) or set(expected_models) != set(model_dirs):
        raise RuntimeError("acceptance model inventory does not cover both reciprocal folds")
    for key, model_dir in model_dirs.items():
        model_path = model_dir / "model.pt"
        actual = sha256_file(model_path)
        if actual != expected_models[key].get("model_sha256"):
            raise RuntimeError(f"accepted model hash mismatch for {key}")
    return payload


def worker_main(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated inference worker must see exactly one GPU, saw {torch.cuda.device_count()}"
        )
    started = time.monotonic()
    requested = tuple(item for item in args.datasets.split(",") if item)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("worker datasets must be non-empty and unique")
    all_videos = rerank.read_submission(args.base_submission)
    if not set(requested).issubset(all_videos):
        raise ValueError("worker was assigned an unknown dataset")
    videos = {stem: all_videos[stem] for stem in requested}
    raw_videos = rerank.read_raw_graphs(args.base_graph_root, set(videos))
    transfer = rerank.transfer_raw_edge_probabilities(videos, raw_videos)
    acceptance = json.loads(args.acceptance_evidence.read_text(encoding="utf-8"))
    selected = acceptance["association_configuration"]

    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    model_dirs = {
        "target_44b6": args.model_44b6_dir,
        "target_6bba": args.model_6bba_dir,
    }
    models: dict[str, torch.nn.Module] = {}
    device = torch.device("cuda:0")
    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    for stem, video in videos.items():
        model_key = MODEL_KEYS[embryo_prefix(stem)]
        if model_key not in models:
            models[model_key] = TrackingTransformer.from_folder(
                model_dirs[model_key], map_location="cpu"
            ).to(device)
            models[model_key].eval()
        edges = rerank.selected_edges_for_video(
            models[model_key],
            video,
            selected,
            device,
            max_tokens=args.max_tokens,
            candidate_radius=args.candidate_radius,
        )
        rerank.validate_edges(video, edges)
        candidate_edges[stem] = edges
        base_set = set(map(tuple, video.edges.tolist()))
        candidate_set = set(edges)
        dataset_stats[stem] = {
            "model_key": model_key,
            "nodes": len(video.node_ids),
            "base_edges": len(base_set),
            "candidate_edges": len(candidate_set),
            "retained_edges": len(base_set & candidate_set),
            "removed_edges": len(base_set - candidate_set),
            "new_edges": len(candidate_set - base_set),
        }
        print(f"worker completed {stem} with {model_key}", flush=True)

    atomic_json(
        args.worker_output,
        {
            "schema_version": 1,
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
            "datasets": list(requested),
            "candidate_edges": {
                stem: [[int(source), int(target)] for source, target in edges]
                for stem, edges in candidate_edges.items()
            },
            "dataset_stats": dataset_stats,
            "edge_probability_transfer": transfer,
        },
    )


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    detected_devices = torch.cuda.device_count()
    if detected_devices != 2:
        raise RuntimeError(f"submission inference requires exactly 2 GPUs, saw {detected_devices}")
    model_dirs = {
        "target_44b6": args.model_44b6_dir,
        "target_6bba": args.model_6bba_dir,
    }
    acceptance = load_acceptance(args.acceptance_evidence, model_dirs)
    videos = rerank.read_submission(args.base_submission)
    weights = {stem: movie_inference_weight(video) for stem, video in videos.items()}
    cuda_tokens = visible_cuda_tokens(detected_devices=detected_devices)
    shards = build_movie_shards(
        sorted(videos), cuda_tokens, movie_weights=weights
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": 1,
        "shard_plan_sha256": shard_plan_sha256(shards),
        "weights": weights,
        "shards": [
            {
                "shard_index": shard.shard_index,
                "cuda_token": shard.cuda_token,
                "movie_ids": list(shard.movie_ids),
                "total_weight": sum(weights[movie] for movie in shard.movie_ids),
            }
            for shard in shards
        ],
    }
    atomic_json(args.output_dir / "shard_plan.json", plan_payload)

    processes: list[tuple[int, subprocess.Popen[str], Any, Path]] = []
    for shard in shards:
        worker_output = args.output_dir / f"shard_{shard.shard_index}.json"
        log_path = args.output_dir / f"shard_{shard.shard_index}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--base-submission", str(args.base_submission),
            "--base-graph-root", str(args.base_graph_root),
            "--model-44b6-dir", str(args.model_44b6_dir),
            "--model-6bba-dir", str(args.model_6bba_dir),
            "--acceptance-evidence", str(args.acceptance_evidence),
            "--trackastra-dir", str(args.trackastra_dir),
            "--output-dir", str(args.output_dir),
            "--datasets", ",".join(shard.movie_ids),
            "--worker-output", str(worker_output),
            "--max-tokens", str(args.max_tokens),
            "--candidate-radius", str(args.candidate_radius),
        ]
        process = subprocess.Popen(
            command,
            env=worker_environment(shard),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard.shard_index, process, log_handle, worker_output))

    return_codes: dict[int, int] = {}
    try:
        while len(return_codes) < len(processes):
            for shard_index, process, _handle, _output in processes:
                code = process.poll()
                if code is not None and shard_index not in return_codes:
                    return_codes[shard_index] = int(code)
            if time.monotonic() - started >= args.hard_stop_seconds:
                raise TimeoutError("two-GPU inference exceeded its hard stop")
            if len(return_codes) < len(processes):
                time.sleep(3)
    finally:
        for _shard_index, process, handle, _output in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
    failures = {index: code for index, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"inference workers failed: {failures}")

    observed_by_shard: dict[int, list[str]] = {}
    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    transfer_reports: dict[str, Any] = {}
    for shard_index, _process, _handle, worker_output in processes:
        payload = json.loads(worker_output.read_text(encoding="utf-8"))
        observed_by_shard[shard_index] = list(payload["datasets"])
        for stem, edges in payload["candidate_edges"].items():
            candidate_edges[stem] = [tuple(map(int, edge)) for edge in edges]
        dataset_stats.update(payload["dataset_stats"])
        transfer_reports[str(shard_index)] = payload["edge_probability_transfer"]
    coverage = validate_shard_outputs(shards, observed_by_shard)
    if set(candidate_edges) != set(videos) or set(coverage) != set(videos):
        raise RuntimeError("candidate edges do not cover every planned movie exactly once")

    output_path = args.output_dir / "submission.csv"
    rerank.write_submission(output_path, videos, candidate_edges)
    total_changed_edges = sum(
        row["removed_edges"] + row["new_edges"] for row in dataset_stats.values()
    )
    if total_changed_edges <= 0:
        raise RuntimeError("dual-fold candidate is an exact base-edge replica")
    report = {
        "schema_version": 1,
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "whole_movie_coverage": list(coverage),
        "shard_plan_sha256": plan_payload["shard_plan_sha256"],
        "base_submission_sha256": sha256_file(args.base_submission),
        "candidate_submission_sha256": sha256_file(output_path),
        "acceptance_evidence_sha256": sha256_file(args.acceptance_evidence),
        "models": acceptance["models"],
        "association_configuration": acceptance["association_configuration"],
        "datasets": dataset_stats,
        "edge_probability_transfer_by_shard": transfer_reports,
        "total_changed_edges": total_changed_edges,
        "nodes_preserved_exactly": True,
        "public_leaderboard_used_for_selection": False,
    }
    atomic_json(args.output_dir / "candidate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--base-submission", type=Path, required=True)
    result.add_argument("--base-graph-root", type=Path, required=True)
    result.add_argument("--model-44b6-dir", type=Path, required=True)
    result.add_argument("--model-6bba-dir", type=Path, required=True)
    result.add_argument("--acceptance-evidence", type=Path, required=True)
    result.add_argument("--trackastra-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--datasets", default="")
    result.add_argument("--worker-output", type=Path)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--candidate-radius", type=float, default=80.0)
    result.add_argument("--hard-stop-seconds", type=int, default=39000)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.worker_output is None:
            raise ValueError("--worker-output is required for a worker")
        worker_main(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
