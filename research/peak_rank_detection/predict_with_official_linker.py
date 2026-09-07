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
import math
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
PEAK_TTA_VIEWS = {"none": 1, "zflip2": 2, "rot4": 4, "d4": 8}
RUNTIME_PROJECTION_SAFETY_FACTOR = 1.15
EXPECTED_ASSOCIATION_CONFIG = {
    "secondary_edge_weight": 0.20,
    "secondary_detection_weight": 0.80,
    "secondary_link_mode": "low_margin_consensus",
    "secondary_mix_temperature": 1.0,
    "secondary_low_margin_max": 0.35,
    "edge_candidate_threshold": 0.48,
    "bidirectional_edge_weight": 0.15,
    "association_coordinate_mode": "subvoxel",
}


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
    training = json.loads(terminal.read_text(encoding="utf-8"))
    validation_path = runtime_root / "clean_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    selected_tta_mode = validation.get("selected_tta_mode")
    ensemble_members = training.get("ensemble_members")
    expected_ensemble_size = (
        len(ensemble_members) if ensemble_members is not None else 1
    )
    ensemble_fusion = training.get("ensemble_fusion")
    if not (
        manifest.get("training_audit_passed") is True
        and isinstance(manifest.get("parameter_count"), int)
        and not isinstance(manifest.get("parameter_count"), bool)
        and manifest.get("parameter_count") == training.get("parameter_count")
        and manifest.get("parameter_count", 0) > 0
        and manifest.get("widths") == training.get("widths")
        and manifest.get("depths") == training.get("depths")
        and manifest.get("ensemble_size") == expected_ensemble_size
        and (
            ensemble_members is None
            or (
                ensemble_fusion
                in {
                    "equal_logit_and_offset_mean",
                    "confidence_max_logit_with_winner_offset",
                }
                and manifest.get("ensemble_fusion") == ensemble_fusion
            )
        )
        and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
        and manifest.get("clean_validation_promotion_passed") is True
        and manifest.get("clean_validation_sha256") == sha256_file(validation_path)
        and manifest.get("selected_peak_tta_mode") == selected_tta_mode
        and selected_tta_mode in PEAK_TTA_VIEWS
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


def load_attributed_association_stack(
    official: Any, weights: Path, device: torch.device
) -> tuple[Any, int, tuple[int, ...], dict[str, Any], dict[str, Any]]:
    """Load both audited public association members with frozen config checks."""

    primary, window_size, downsample = official.load_model(weights, device)
    secondary_text = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()
    if not secondary_text:
        raise RuntimeError("audited secondary association weights are not configured")
    secondary_path = Path(secondary_text)
    if not secondary_path.is_file():
        raise FileNotFoundError(secondary_path)
    numeric_environment = {
        "secondary_edge_weight": "BIOHUB_SECONDARY_EDGE_WEIGHT",
        "secondary_detection_weight": "BIOHUB_SECONDARY_DETECTION_WEIGHT",
        "secondary_mix_temperature": "BIOHUB_SECONDARY_MIX_TEMPERATURE",
        "secondary_low_margin_max": "BIOHUB_SECONDARY_LOW_MARGIN_MAX",
        "edge_candidate_threshold": "BIOHUB_DUAL_SEED_EDGE_THRESHOLD",
        "bidirectional_edge_weight": "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT",
    }
    observed: dict[str, Any] = {
        name: float(os.environ.get(environment, "nan"))
        for name, environment in numeric_environment.items()
    }
    observed["secondary_link_mode"] = os.environ.get(
        "BIOHUB_SECONDARY_LINK_MODE", ""
    ).strip()
    observed["association_coordinate_mode"] = "subvoxel"
    for name, expected in EXPECTED_ASSOCIATION_CONFIG.items():
        actual = observed[name]
        if isinstance(expected, float):
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"audited association config drift for {name}: {actual} != {expected}"
                )
        elif actual != expected:
            raise ValueError(
                f"audited association config drift for {name}: {actual!r} != {expected!r}"
            )
    if os.environ.get("BIOHUB_EDGE_FEATURE_TTA", "") != "1":
        raise ValueError("audited association feature TTA is not enabled")
    secondary, secondary_window, secondary_downsample = official.load_model(
        secondary_path, device
    )
    if secondary_window != window_size or secondary_downsample != downsample:
        raise ValueError("primary and secondary association grids differ")
    kwargs = {
        "secondary_model": secondary,
        "secondary_edge_weight": observed["secondary_edge_weight"],
        # External detections own the nodes. This remains nonzero because the
        # audited source gates secondary feature TTA behind this setting.
        "secondary_detection_weight": observed["secondary_detection_weight"],
        "secondary_link_mode": observed["secondary_link_mode"],
        "secondary_mix_temperature": observed["secondary_mix_temperature"],
        "secondary_low_margin_max": observed["secondary_low_margin_max"],
        "association_coordinate_mode": observed["association_coordinate_mode"],
    }
    manifest = {
        **observed,
        "primary_weights_sha256": sha256_file(weights),
        "secondary_weights_sha256": sha256_file(secondary_path),
        "edge_feature_tta": True,
    }
    return primary, window_size, downsample, kwargs, manifest


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
    selected_tta_mode = str(runtime_manifest["selected_peak_tta_mode"])
    if args.peak_tta_mode != selected_tta_mode:
        raise RuntimeError(
            f"requested peak TTA {args.peak_tta_mode} differs from clean-selected "
            f"mode {selected_tta_mode}"
        )
    official = import_official_predictor(args.official_predictor)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("production worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    training_terminal = validate_training(checkpoint, terminal_path)
    detector = load_model(checkpoint, training_terminal, device)
    (
        association_model,
        window_size,
        downsample,
        association_kwargs,
        association_manifest,
    ) = load_attributed_association_stack(official, args.weights, device)
    cfg = official.PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )
    cfg.threshold = association_manifest["edge_candidate_threshold"]
    worker_index, worker_count = resolved_worker_identity(args)
    output_dir = args.output_dir
    if output_dir is None:
        from dataspec import PREDICTIONS_PATH

        output_dir = (
            Path(PREDICTIONS_PATH)
            / official.USERNAME
            / args.method
            / f"split_{args.fold}"
        )
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
    import zarr

    frame_counts = {}
    for name in names:
        path = sample_path_for_name(args.data_dir, name)
        if path is None:
            raise FileNotFoundError(name)
        frame_counts[name] = int(zarr.open_group(str(path), mode="r")["0"].shape[0])
    total_frame_count = sum(frame_counts.values())
    rows = []
    projected_worker_seconds = 0.0
    for name in names:
        movie_started = time.monotonic()
        sample_path = sample_path_for_name(args.data_dir, name)
        if sample_path is None:
            raise FileNotFoundError(
                f"movie disappeared after inventory validation: {name}"
            )
        cache = predict_movie_detection_cache(
            detector,
            sample_path,
            device=device,
            batch_size=args.peak_batch_size,
            calibration_frames=args.calibration_frames,
            tta_mode=args.peak_tta_mode,
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
            **association_kwargs,
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
        movie_elapsed = time.monotonic() - movie_started
        rows.append(
            {
                "dataset": name.removesuffix(".zarr"),
                "nodes": int(graph.num_nodes()),
                "raw_edges": raw_edges,
                "ilp_edges": int(graph.num_edges()),
                "density_threshold": cache.threshold,
                "estimated_node_count": cache.estimated_node_count,
                "projected_node_count": cache.projected_node_count,
                "frame_count": frame_counts[name],
                "elapsed_seconds": movie_elapsed,
                "seconds_per_frame": movie_elapsed / frame_counts[name],
                "peak_tta_mode": selected_tta_mode,
                "peak_tta_views": PEAK_TTA_VIEWS[selected_tta_mode],
            }
        )
        print("PEAK TRACKING MOVIE", json.dumps(rows[-1], sort_keys=True), flush=True)
        completed_seconds = sum(float(row["elapsed_seconds"]) for row in rows)
        completed_frames = sum(int(row["frame_count"]) for row in rows)
        projected_worker_seconds = (
            completed_seconds
            * total_frame_count
            / completed_frames
            * RUNTIME_PROJECTION_SAFETY_FACTOR
        )
        print(
            "PEAK TRACKING RUNTIME",
            json.dumps(
                {
                    "completed_movies": len(rows),
                    "worker_movies": len(names),
                    "completed_frames": completed_frames,
                    "worker_frames": total_frame_count,
                    "safety_factor": RUNTIME_PROJECTION_SAFETY_FACTOR,
                    "projected_worker_seconds": projected_worker_seconds,
                    "worker_budget_seconds": args.max_projected_worker_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if projected_worker_seconds > args.max_projected_worker_seconds:
            raise TimeoutError(
                "measured peak-ranking throughput cannot finish inside the frozen "
                f"worker budget: {projected_worker_seconds:.1f}s projected"
            )
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
            "peak_tta_mode": selected_tta_mode,
            "peak_tta_views": PEAK_TTA_VIEWS[selected_tta_mode],
            "total_frame_count": total_frame_count,
            "movie_compute_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "worker_elapsed_seconds": time.monotonic() - started,
            "projected_worker_seconds": projected_worker_seconds,
            "runtime_projection_safety_factor": RUNTIME_PROJECTION_SAFETY_FACTOR,
            "worker_budget_seconds": args.max_projected_worker_seconds,
            "association": association_manifest,
            "checkpoint_sha256": runtime_manifest["checkpoint_sha256"],
            "parameter_count": runtime_manifest["parameter_count"],
            "ensemble_size": runtime_manifest["ensemble_size"],
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
    parser.add_argument("--peak-tta-mode", choices=tuple(PEAK_TTA_VIEWS), required=True)
    parser.add_argument("--max-projected-worker-seconds", type=float, default=31_500.0)
    parser.add_argument("--max-wall-seconds", type=float, default=39_000.0)
    args = parser.parse_args()
    if (
        args.peak_batch_size <= 0
        or args.unet_batch_size <= 0
        or args.calibration_frames <= 0
        or args.max_wall_seconds <= 0
        or args.max_projected_worker_seconds <= 0
    ):
        parser.error("batch, calibration, and wall limits must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
