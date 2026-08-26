#!/usr/bin/env python
"""Evaluate official Spotiflow 3D checkpoints on clean Biohub fields.

Model/normalization selection uses eight fields disjoint from the four complete
acceptance movies.  Within each movie the organizer-provided node-count estimate
sets the detector threshold using image predictions only; ground truth is read
after threshold selection and is used only for reporting annotated-node recall.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from density_calibration import read_estimated_node_count, uniform_frame_indices
except ModuleNotFoundError:  # Imported as ``research.spotiflow_biohub`` in tests.
    from research.density_calibration import (
        read_estimated_node_count,
        uniform_frame_indices,
    )


SCREEN_STEMS = (
    "44b6_d29c9ab2",
    "44b6_3a861e03",
    "44b6_d5e7d891",
    "44b6_ddf577ad",
    "6bba_09961292",
    "6bba_bb9f20c3",
    "6bba_784a78c9",
    "6bba_57b7cc1e",
)
ACCEPTANCE_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
SPATIAL_DOWNSAMPLE = np.asarray((1.0, 4.0, 4.0), dtype=np.float64)
VOXEL_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
MATCH_RADIUS_UM = 5.0
LOW_PROBABILITY_THRESHOLD = 0.05
EXPECTED_PARAMETER_COUNT = 35_489_892


@dataclass(frozen=True)
class FramePeaks:
    frame: int
    points_input: np.ndarray
    probabilities: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_from_geff(path: Path):
    import tracksdata as td

    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def graph_points_by_frame(path: Path) -> dict[int, np.ndarray]:
    graph = graph_from_geff(path)
    result: dict[int, list[tuple[float, float, float]]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        result.setdefault(int(row["t"]), []).append(
            (float(row["z"]), float(row["y"]), float(row["x"]))
        )
    return {
        frame: np.asarray(points, dtype=np.float64).reshape(-1, 3)
        for frame, points in result.items()
    }


def density_threshold(
    sampled: Sequence[FramePeaks], estimated_count: float, n_frames: int
) -> tuple[float, float]:
    """Choose a strict probability threshold closest to the metadata count."""

    if not sampled:
        raise ValueError("sampled predictions cannot be empty")
    pooled = np.concatenate([frame.probabilities for frame in sampled])
    desired = int(round(estimated_count * len(sampled) / n_frames))
    if len(pooled) == 0:
        return LOW_PROBABILITY_THRESHOLD, 0.0
    desired = max(1, min(desired, len(pooled)))
    ordered = np.sort(pooled)[::-1]
    boundary = float(ordered[desired - 1])
    if desired < len(ordered):
        threshold = (boundary + float(ordered[desired])) / 2.0
    else:
        threshold = float(np.nextafter(LOW_PROBABILITY_THRESHOLD, 0.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    selected = sum(int(np.count_nonzero(frame.probabilities > threshold)) for frame in sampled)
    projected = float(selected) * n_frames / len(sampled)
    return threshold, projected


def match_frame(
    predicted_input: np.ndarray, gt_voxel: np.ndarray
) -> tuple[int, int, list[float]]:
    """One-to-one physical-distance matching against annotated GT nodes."""

    predicted_input = np.asarray(predicted_input, dtype=np.float64).reshape(-1, 3)
    gt_voxel = np.asarray(gt_voxel, dtype=np.float64).reshape(-1, 3)
    if len(gt_voxel) == 0:
        return 0, 0, []
    if len(predicted_input) == 0:
        return 0, len(gt_voxel), []
    predicted_um = predicted_input * SPATIAL_DOWNSAMPLE * VOXEL_SCALE_UM
    gt_um = gt_voxel * VOXEL_SCALE_UM
    distances = np.linalg.norm(
        predicted_um[:, None, :] - gt_um[None, :, :], axis=-1
    )
    rows, cols = linear_sum_assignment(distances)
    accepted = [float(distances[row, col]) for row, col in zip(rows, cols) if distances[row, col] <= MATCH_RADIUS_UM]
    return len(accepted), len(gt_voxel), accepted


def score_predictions(
    predictions: Sequence[FramePeaks],
    gt_by_frame: dict[int, np.ndarray],
    threshold: float,
) -> dict[str, float | int]:
    matched = 0
    annotated = 0
    distances: list[float] = []
    counts: list[int] = []
    for frame in predictions:
        keep = frame.probabilities > threshold
        points = frame.points_input[keep]
        counts.append(len(points))
        frame_matched, frame_annotated, frame_distances = match_frame(
            points, gt_by_frame.get(frame.frame, np.empty((0, 3)))
        )
        matched += frame_matched
        annotated += frame_annotated
        distances.extend(frame_distances)
    return {
        "matched_gt_nodes": matched,
        "annotated_gt_nodes": annotated,
        "annotated_node_recall": matched / annotated if annotated else 0.0,
        "mean_match_distance_um": float(np.mean(distances)) if distances else float("inf"),
        "predicted_nodes": int(sum(counts)),
        "temporal_count_roughness": (
            float(np.mean(np.abs(np.diff(counts)))) / max(float(np.mean(counts)), 1.0)
            if len(counts) > 1
            else 0.0
        ),
    }


def score_graph_nodes(
    pred_by_frame: dict[int, np.ndarray], gt_by_frame: dict[int, np.ndarray], n_frames: int
) -> dict[str, float | int]:
    predictions = [
        FramePeaks(
            frame=frame,
            points_input=np.asarray(pred_by_frame.get(frame, np.empty((0, 3))))
            / SPATIAL_DOWNSAMPLE,
            probabilities=np.ones(len(pred_by_frame.get(frame, ())), dtype=np.float64),
        )
        for frame in range(n_frames)
    ]
    return score_predictions(predictions, gt_by_frame, 0.5)


def load_image_context(sample_path: Path):
    import zarr
    from biohub_tracking.io import open_dataset

    dataset = open_dataset(
        sample_path, normalize=False, load_image=False, require_tracks=False
    )
    array = zarr.open_group(str(dataset.zarr_path), mode="r")["0"]
    if "0.001" not in dataset.quantiles or "0.999" not in dataset.quantiles:
        raise ValueError(f"Missing image quantiles for {sample_path}")
    return (
        dataset,
        array,
        float(dataset.quantiles["0.001"]),
        float(dataset.quantiles["0.999"]),
    )


def predict_frames(
    model,
    sample_path: Path,
    frames: Iterable[int],
    normalization: str,
) -> tuple[list[FramePeaks], int]:
    dataset, array, q_low, q_high = load_image_context(sample_path)
    n_frames = int(dataset.image_shape[0])
    results: list[FramePeaks] = []
    for frame in frames:
        raw = array[int(frame), :, ::4, ::4].astype(np.float32)
        if raw.shape != (64, 64, 64):
            raise ValueError(f"Unexpected downsampled shape for {sample_path}: {raw.shape}")
        if normalization == "biohub_global":
            image = np.clip((raw - q_low) / (q_high - q_low + 1e-6), 0.0, None)
            normalizer = None
        elif normalization == "spotiflow_auto":
            image = raw
            normalizer = "auto"
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        try:
            points, details = model.predict(
                image,
                prob_thresh=LOW_PROBABILITY_THRESHOLD,
                n_tiles=(1, 1, 1),
                min_distance=1,
                exclude_border=False,
                subpix=True,
                normalizer=normalizer,
                verbose=False,
                device="cuda",
            )
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            import torch

            torch.cuda.empty_cache()
            points, details = model.predict(
                image,
                prob_thresh=LOW_PROBABILITY_THRESHOLD,
                n_tiles=(2, 2, 2),
                min_distance=1,
                exclude_border=False,
                subpix=True,
                normalizer=normalizer,
                verbose=False,
                device="cuda",
            )
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        probabilities = np.asarray(details.prob, dtype=np.float64).reshape(-1)
        if len(points) != len(probabilities):
            raise RuntimeError(
                f"Spotiflow returned {len(points)} points but {len(probabilities)} probabilities"
            )
        results.append(FramePeaks(int(frame), points, probabilities))
    return results, n_frames


def evaluate_variant(
    model,
    competition_dir: Path,
    normalization: str,
    stems: Sequence[str],
    sample_frames: int,
) -> dict:
    rows = []
    started = time.monotonic()
    for stem in stems:
        sample_path = competition_dir / "train" / f"{stem}.zarr"
        truth_path = competition_dir / "train" / f"{stem}.geff"
        gt_by_frame = graph_points_by_frame(truth_path)
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"Missing estimated_number_of_nodes in {truth_path}")
        dataset, _, _, _ = load_image_context(sample_path)
        frames = uniform_frame_indices(int(dataset.image_shape[0]), sample_frames)
        predictions, n_frames = predict_frames(model, sample_path, frames, normalization)
        threshold, projected = density_threshold(predictions, estimated, n_frames)
        score = score_predictions(predictions, gt_by_frame, threshold)
        rows.append(
            {
                "stem": stem,
                "estimated_node_count": estimated,
                "threshold": threshold,
                "projected_node_count": projected,
                "projected_count_ratio": projected / estimated,
                **score,
            }
        )
        print("SCREEN", normalization, json.dumps(rows[-1], sort_keys=True), flush=True)
    annotated = sum(int(row["annotated_gt_nodes"]) for row in rows)
    matched = sum(int(row["matched_gt_nodes"]) for row in rows)
    finite_distances = [
        float(row["mean_match_distance_um"])
        for row in rows
        if np.isfinite(float(row["mean_match_distance_um"]))
    ]
    return {
        "normalization": normalization,
        "rows": rows,
        "annotated_gt_nodes": annotated,
        "matched_gt_nodes": matched,
        "annotated_node_recall": matched / annotated if annotated else 0.0,
        "mean_movie_match_distance_um": float(np.mean(finite_distances)),
        "mean_abs_log_count_ratio": float(
            np.mean([abs(np.log(max(float(row["projected_count_ratio"]), 1e-8))) for row in rows])
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def choose_variant(results: Sequence[dict]) -> dict:
    return max(
        results,
        key=lambda row: (
            float(row["annotated_node_recall"]),
            -float(row["mean_movie_match_distance_um"]),
            -float(row["mean_abs_log_count_ratio"]),
            row["model_name"],
            row["normalization"],
        ),
    )


def main(args: argparse.Namespace) -> None:
    import torch
    from spotiflow.model import Spotiflow

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Spotiflow detector acceptance run")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "synth_3d": args.synth_model,
        "smfish_3d": args.smfish_model,
    }
    provenance = {}
    selection_rows = []
    for model_name, model_dir in model_dirs.items():
        weight_path = model_dir / "best.pt"
        provenance[model_name] = {
            "model_dir": str(model_dir),
            "best_sha256": sha256_file(weight_path),
        }
        model = Spotiflow.from_folder(
            str(model_dir), which="best", inference_mode=True, map_location="cuda"
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"Unexpected {model_name} parameter count: {parameter_count}"
            )
        for normalization in ("biohub_global", "spotiflow_auto"):
            result = evaluate_variant(
                model,
                args.competition_dir,
                normalization,
                SCREEN_STEMS,
                args.screen_frames,
            )
            result["model_name"] = model_name
            result["parameter_count"] = parameter_count
            selection_rows.append(result)
            (args.output_dir / "selection_partial.json").write_text(
                json.dumps(selection_rows, indent=2, sort_keys=True), encoding="utf-8"
            )
            if time.monotonic() - started > args.max_wall_seconds:
                raise TimeoutError("Wall-clock guard reached during model selection")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    selected = choose_variant(selection_rows)
    selected_name = str(selected["model_name"])
    selected_normalization = str(selected["normalization"])
    print(
        "SELECTED",
        json.dumps(
            {
                "model_name": selected_name,
                "normalization": selected_normalization,
                "annotated_node_recall": selected["annotated_node_recall"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model = Spotiflow.from_folder(
        str(model_dirs[selected_name]),
        which="best",
        inference_mode=True,
        map_location="cuda",
    )
    acceptance_rows = []
    for stem in ACCEPTANCE_STEMS:
        sample_path = args.competition_dir / "train" / f"{stem}.zarr"
        truth_path = args.competition_dir / "train" / f"{stem}.geff"
        baseline_path = args.baseline_predictions / f"{stem}.geff"
        gt_by_frame = graph_points_by_frame(truth_path)
        baseline_by_frame = graph_points_by_frame(baseline_path)
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"Missing estimated count for {stem}")
        dataset, _, _, _ = load_image_context(sample_path)
        n_frames = int(dataset.image_shape[0])
        predictions, _ = predict_frames(
            model, sample_path, range(n_frames), selected_normalization
        )
        sampled_indices = set(uniform_frame_indices(n_frames, args.calibration_frames).tolist())
        sampled_predictions = [row for row in predictions if row.frame in sampled_indices]
        threshold, projected = density_threshold(
            sampled_predictions, estimated, n_frames
        )
        score = score_predictions(predictions, gt_by_frame, threshold)
        baseline_score = score_graph_nodes(baseline_by_frame, gt_by_frame, n_frames)
        row = {
            "stem": stem,
            "estimated_node_count": estimated,
            "threshold": threshold,
            "calibration_projected_node_count": projected,
            "calibration_projected_count_ratio": projected / estimated,
            "spotiflow": score,
            "baseline_raw_graph": baseline_score,
            "annotated_recall_delta": float(score["annotated_node_recall"])
            - float(baseline_score["annotated_node_recall"]),
            "final_count_ratio": float(score["predicted_nodes"]) / estimated,
            "baseline_count_ratio": float(baseline_score["predicted_nodes"]) / estimated,
        }
        acceptance_rows.append(row)
        print("ACCEPTANCE", json.dumps(row, sort_keys=True), flush=True)
        (args.output_dir / "acceptance_partial.json").write_text(
            json.dumps(acceptance_rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("Wall-clock guard reached during acceptance")

    spot_annotated = sum(int(row["spotiflow"]["annotated_gt_nodes"]) for row in acceptance_rows)
    spot_matched = sum(int(row["spotiflow"]["matched_gt_nodes"]) for row in acceptance_rows)
    base_matched = sum(int(row["baseline_raw_graph"]["matched_gt_nodes"]) for row in acceptance_rows)
    summary = {
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "selection": {
            "model_name": selected_name,
            "normalization": selected_normalization,
            "annotated_node_recall": selected["annotated_node_recall"],
        },
        "acceptance": {
            "annotated_gt_nodes": spot_annotated,
            "spotiflow_matched_gt_nodes": spot_matched,
            "baseline_matched_gt_nodes": base_matched,
            "spotiflow_annotated_node_recall": spot_matched / spot_annotated,
            "baseline_annotated_node_recall": base_matched / spot_annotated,
            "annotated_recall_delta": (spot_matched - base_matched) / spot_annotated,
            "rows": acceptance_rows,
        },
        "provenance": provenance,
        "screen_stems": list(SCREEN_STEMS),
        "acceptance_stems": list(ACCEPTANCE_STEMS),
        "selection_acceptance_overlap": sorted(set(SCREEN_STEMS) & set(ACCEPTANCE_STEMS)),
        "parameters": {
            "screen_frames": args.screen_frames,
            "calibration_frames": args.calibration_frames,
            "low_probability_threshold": LOW_PROBABILITY_THRESHOLD,
            "match_radius_um": MATCH_RADIUS_UM,
            "spatial_downsample": SPATIAL_DOWNSAMPLE.tolist(),
        },
    }
    (args.output_dir / "detector_acceptance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TERMINAL", json.dumps(summary["acceptance"], sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--synth-model", type=Path, required=True)
    parser.add_argument("--smfish-model", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen-frames", type=int, default=12)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=6600.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
