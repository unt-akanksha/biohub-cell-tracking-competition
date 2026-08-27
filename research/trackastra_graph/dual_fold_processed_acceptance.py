"""Materialize one predeclared dual-fold candidate on the processed control.

This is GPU inference only.  It does not read ground-truth labels, tune a
threshold, run the official scorer, promote a model, create a competition zip,
or submit.  Exact scoring is a separate CPU step after this configuration has
been frozen and materialized once.
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

import torch

try:
    import rerank_submission as rerank
except ModuleNotFoundError:
    from research.trackastra_graph import rerank_submission as rerank


RUN_ID = "trackastra-dual-fold-processed-acceptance-v1"
EXPECTED_STEMS = frozenset(
    {
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)
FOLD_BY_PREFIX = {"44b6": "target_44b6", "6bba": "target_6bba"}
FROZEN_ASSOCIATION_CONFIGURATION: dict[str, Any] = {
    "method": "raw_confidence_hybrid",
    "edge_threshold": 0.08,
    "base_lock_probability": 0.98,
    "base_keep_probability": 0.80,
    "base_bonus": 0.05,
    "division_threshold": 0.18,
    "division_ratio": 0.50,
    "base_division_keep_probability": 0.95,
    "base_pseudo_probability": 0.80,
}


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


def configuration_sha256(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_training_output(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    terminal_path = root / "training_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "completed" or terminal.get("gpu_count") != 2:
        raise RuntimeError("source dual-fold training is not complete")
    if terminal.get("submission_created") is not False:
        raise RuntimeError("source training unexpectedly created a submission")
    if terminal.get("both_folds_improved") is not True:
        raise RuntimeError("both reciprocal folds must improve before processed acceptance opens")
    folds = terminal.get("folds", {})
    if set(folds) != set(FOLD_BY_PREFIX.values()):
        raise RuntimeError("source terminal does not cover both reciprocal folds")
    model_dirs = {fold: root / fold for fold in folds}
    for fold, model_dir in model_dirs.items():
        model_path = model_dir / "model.pt"
        if sha256_file(model_path) != folds[fold].get("model_sha256"):
            raise RuntimeError(f"source model hash mismatch for {fold}")
        if int(folds[fold].get("best_step", 0)) <= 0:
            raise RuntimeError(f"source fold did not beat initialization: {fold}")
    return terminal, model_dirs


def worker_main(args: argparse.Namespace) -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"acceptance worker must see one GPU, saw {torch.cuda.device_count()}")
    started = time.monotonic()
    requested = tuple(item for item in args.datasets.split(",") if item)
    videos = rerank.read_submission(args.processed_control_csv)
    if set(requested) != {stem for stem in EXPECTED_STEMS if stem.startswith(args.prefix)}:
        raise RuntimeError("worker prefix coverage is not the frozen two-movie set")
    videos = {stem: videos[stem] for stem in requested}
    raw_videos = rerank.read_raw_graphs(args.raw_graph_root, set(videos))
    transfer = rerank.transfer_raw_edge_probabilities(videos, raw_videos)

    sys.path.insert(0, str(args.trackastra_dir.resolve()))
    from trackastra.model.model import TrackingTransformer

    device = torch.device("cuda:0")
    model = TrackingTransformer.from_folder(args.model_dir, map_location="cpu").to(device)
    model.eval()
    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    for stem, video in videos.items():
        edges = rerank.selected_edges_for_video(
            model,
            video,
            FROZEN_ASSOCIATION_CONFIGURATION,
            device,
            max_tokens=args.max_tokens,
            candidate_radius=args.candidate_radius,
        )
        rerank.validate_edges(video, edges)
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
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"processed acceptance requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    source_terminal, model_dirs = verify_training_output(args.training_output_root)
    videos = rerank.read_submission(args.processed_control_csv)
    if set(videos) != EXPECTED_STEMS:
        raise RuntimeError("processed control does not contain the frozen four movies")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
    for gpu_index, prefix in enumerate(sorted(FOLD_BY_PREFIX)):
        fold = FOLD_BY_PREFIX[prefix]
        datasets = sorted(stem for stem in EXPECTED_STEMS if stem.startswith(prefix))
        worker_output = args.output_dir / f"{fold}.json"
        log_path = args.output_dir / f"{fold}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--processed-control-csv", str(args.processed_control_csv),
            "--raw-graph-root", str(args.raw_graph_root),
            "--training-output-root", str(args.training_output_root),
            "--trackastra-dir", str(args.trackastra_dir),
            "--output-dir", str(args.output_dir),
            "--model-dir", str(model_dirs[fold]),
            "--prefix", prefix,
            "--datasets", ",".join(datasets),
            "--worker-output", str(worker_output),
            "--max-tokens", str(args.max_tokens),
            "--candidate-radius", str(args.candidate_radius),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((fold, process, log_handle, worker_output))

    return_codes: dict[str, int] = {}
    try:
        while len(return_codes) < len(processes):
            for fold, process, _handle, _output in processes:
                code = process.poll()
                if code is not None and fold not in return_codes:
                    return_codes[fold] = int(code)
            if time.monotonic() - started >= args.hard_stop_seconds:
                raise TimeoutError("processed acceptance exceeded its hard stop")
            if len(return_codes) < len(processes):
                time.sleep(3)
    finally:
        for _fold, process, handle, _output in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"processed acceptance workers failed: {failures}")

    candidate_edges: dict[str, list[tuple[int, int]]] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    transfers: dict[str, Any] = {}
    for fold, _process, _handle, output in processes:
        payload = json.loads(output.read_text(encoding="utf-8"))
        for stem, edges in payload["candidate_edges"].items():
            if stem in candidate_edges:
                raise RuntimeError(f"duplicate movie from acceptance workers: {stem}")
            candidate_edges[stem] = [tuple(map(int, edge)) for edge in edges]
        dataset_stats.update(payload["dataset_stats"])
        transfers[fold] = payload["edge_probability_transfer"]
    if set(candidate_edges) != EXPECTED_STEMS:
        raise RuntimeError("processed candidate does not cover all four movies exactly once")

    candidate_csv = args.output_dir / "processed_candidate.csv"
    rerank.write_submission(candidate_csv, videos, candidate_edges)
    changed_edges = sum(
        row["removed_edges"] + row["new_edges"] for row in dataset_stats.values()
    )
    if changed_edges <= 0 or sha256_file(candidate_csv) == sha256_file(args.processed_control_csv):
        raise RuntimeError("processed candidate is an exact control replica")
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "evaluation_kind": "predeclared_processed_candidate_materialization",
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "whole_movie_sharding": True,
        "processed_control_sha256": sha256_file(args.processed_control_csv),
        "processed_candidate_sha256": sha256_file(candidate_csv),
        "source_training_terminal_sha256": sha256_file(
            args.training_output_root / "training_terminal.json"
        ),
        "models": {
            fold: {
                "model_sha256": source_terminal["folds"][fold]["model_sha256"],
                "best_step": source_terminal["folds"][fold]["best_step"],
            }
            for fold in sorted(model_dirs)
        },
        "association_configuration": FROZEN_ASSOCIATION_CONFIGURATION,
        "association_configuration_sha256": configuration_sha256(
            FROZEN_ASSOCIATION_CONFIGURATION
        ),
        "configuration_selection": "predeclared historical raw-confidence preset; this acceptance run performs no configuration search",
        "datasets": dataset_stats,
        "edge_probability_transfer": transfers,
        "total_changed_edges": changed_edges,
        "ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
        "exact_processed_scoring_performed": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_dir / "materialization_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--processed-control-csv", type=Path, required=True)
    result.add_argument("--raw-graph-root", type=Path, required=True)
    result.add_argument("--training-output-root", type=Path, required=True)
    result.add_argument("--trackastra-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--prefix", choices=sorted(FOLD_BY_PREFIX))
    result.add_argument("--datasets", default="")
    result.add_argument("--worker-output", type=Path)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--candidate-radius", type=float, default=80.0)
    result.add_argument("--hard-stop-seconds", type=int, default=10800)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.model_dir is None or args.prefix is None or args.worker_output is None:
            raise ValueError("worker requires model, prefix, and output arguments")
        worker_main(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
