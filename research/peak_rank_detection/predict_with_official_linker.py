#!/usr/bin/env python
"""Run peak-ranking nodes through an audited official Trackastra linker.

This is a label-free production worker. Multiple workers receive disjoint
movie slices and write separate graph directories; a notebook controller may
merge them only after every worker exits successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

try:
    from evaluate_peak_rank_detector import load_model, validate_training
    from peak_association_bridge import predict_movie_detection_cache
    from lsm_association_bridge import predict_video_with_external_detections
except ModuleNotFoundError:
    from research.peak_rank_detection.evaluate_peak_rank_detector import (
        load_model,
        validate_training,
    )
    from research.peak_rank_detection.association_bridge import (
        predict_movie_detection_cache,
    )
    from research.lsm_fm_detection.association_bridge import (
        predict_video_with_external_detections,
    )


RUN_ID = "peak-rank-official-linker-production-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_runtime(runtime_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = runtime_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = runtime_root / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"peak-ranking runtime file changed: {name}")
    checkpoint = runtime_root / "peak_rank_detector.pt"
    terminal = runtime_root / "training_terminal.json"
    if not (
        manifest.get("training_audit_passed") is True
        and manifest.get("parameter_count") == 38_381_478
        and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
    ):
        raise RuntimeError("peak-ranking runtime manifest is ineligible")
    return checkpoint, terminal, manifest


def import_official_predictor(source: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("biohub_official_predict", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official predictor from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in (
        "PredictConfig",
        "load_model",
        "predict_video",
        "_detect_cells_pooled",
        "build_graph",
        "save_graph",
        "td",
    ):
        if not hasattr(module, required):
            raise RuntimeError(f"official predictor lacks {required}")
    return module


def selected_names(
    data_dir: Path,
    splits_path: Path,
    fold: int,
    worker_index: int,
    worker_count: int,
    video_slice: slice | None = None,
) -> list[str]:
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker partition")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    names = [str(name) for name in splits[fold]["test"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate test movie in split")
    missing = [name for name in names if sample_path_for_name(data_dir, name) is None]
    if missing:
        raise FileNotFoundError(f"missing test movies: {missing}")
    if video_slice is not None:
        return names[video_slice]
    return names[worker_index::worker_count]


def sample_path_for_name(data_dir: Path, name: str) -> Path | None:
    for candidate in (data_dir / name, data_dir / f"{name}.zarr"):
        if candidate.is_dir():
            return candidate
    return None


def parse_slice(value: str | None) -> slice | None:
    if value is None:
        return None
    fields = value.split(":")
    if not 1 <= len(fields) <= 3:
        raise ValueError(f"invalid movie slice: {value}")
    fields.extend([""] * (3 - len(fields)))
    return slice(*(int(field) if field else None for field in fields))


def resolved_worker_identity(args: argparse.Namespace) -> tuple[int, int]:
    if args.worker_index is not None:
        return int(args.worker_index), int(args.worker_count)
    shard = os.environ.get("BIOHUB_GPU_SHARD", "").strip()
    if shard:
        index, count = shard.split("/", maxsplit=1)
        return int(index), int(count)
    return 0, 1


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    checkpoint, terminal_path, runtime_manifest = verify_runtime(args.runtime_root)
    official = import_official_predictor(args.official_predictor)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("production worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    training_terminal = validate_training(checkpoint, terminal_path)
    detector = load_model(checkpoint, training_terminal, device)
    association_model, window_size, downsample = official.load_model(
        args.weights, device
    )
    cfg = official.PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
    worker_index, worker_count = resolved_worker_identity(args)
    output_dir = args.output_dir
    if output_dir is None:
        from dataspec import PREDICTIONS_PATH

        output_dir = Path(PREDICTIONS_PATH) / official.USERNAME / args.method / f"split_{args.fold}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse production output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    names = selected_names(
        args.data_dir,
        args.splits,
        args.fold,
        worker_index,
        worker_count,
        parse_slice(args.video_slice),
    )
    rows = []
    for name in names:
        sample_path = sample_path_for_name(args.data_dir, name)
        if sample_path is None:
            raise FileNotFoundError(f"movie disappeared after inventory validation: {name}")
        cache = predict_movie_detection_cache(
            detector,
            sample_path,
            device=device,
            batch_size=args.peak_batch_size,
            calibration_frames=args.calibration_frames,
            d4_tta=not args.disable_peak_d4_tta,
        )
        coordinates, edges = predict_video_with_external_detections(
            official,
            association_model,
            sample_path,
            device,
            cfg,
            cache,
            window_size=window_size,
            unet_batch_size=args.unet_batch_size,
            downsample=downsample,
        )
        graph = official.build_graph(coordinates, edges)
        raw_edges = int(graph.num_edges())
        if cfg.use_ilp and raw_edges > 0:
            solver = official.td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * official.td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            with official.suppress_output():
                graph = solver.solve(graph)
        output = output_dir / f"{name}.geff"
        official.save_graph(graph, output)
        rows.append(
            {
                "dataset": name.removesuffix(".zarr"),
                "nodes": int(graph.num_nodes()),
                "raw_edges": raw_edges,
                "ilp_edges": int(graph.num_edges()),
                "density_threshold": cache.threshold,
                "estimated_node_count": cache.estimated_node_count,
                "projected_node_count": cache.projected_node_count,
            }
        )
        print("PEAK TRACKING MOVIE", json.dumps(rows[-1], sort_keys=True), flush=True)
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("peak-ranking production worker exceeded its wall guard")
    atomic_json(
        output_dir / "worker_manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "worker_index": worker_index,
            "worker_count": worker_count,
            "movies": rows,
            "checkpoint_sha256": runtime_manifest["checkpoint_sha256"],
            "input_partition": args.data_dir.name,
            "competition_train_labels_read": False,
            "competition_test_labels_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--official-predictor", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", "--split", dest="fold", type=int, default=0)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", default="unet_transformer")
    parser.add_argument("--slice", dest="video_slice")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--peak-batch-size", type=int, default=1)
    parser.add_argument("--unet-batch-size", type=int, default=4)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--det-threshold", type=float, default=0.965)
    parser.add_argument("--use-ilp", action="store_true")
    parser.add_argument("--ilp-edge-weight", type=float, default=-1.0)
    parser.add_argument("--ilp-appearance-weight", type=float, default=0.0)
    parser.add_argument("--ilp-disappearance-weight", type=float, default=2.0)
    parser.add_argument("--ilp-division-weight", type=float, default=1.2)
    parser.add_argument("--disable-peak-d4-tta", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=39_000.0)
    args = parser.parse_args()
    if (
        args.peak_batch_size <= 0
        or args.unet_batch_size <= 0
        or args.calibration_frames <= 0
        or args.max_wall_seconds <= 0
    ):
        parser.error("batch, calibration, and wall limits must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
