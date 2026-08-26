#!/usr/bin/env python
"""Evaluate one learned Spotiflow checkpoint on disjoint complete Biohub movies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        EXPECTED_PARAMETER_COUNT,
        LOW_PROBABILITY_THRESHOLD,
        MATCH_RADIUS_UM,
        SPATIAL_DOWNSAMPLE,
        density_threshold,
        graph_points_by_frame,
        load_image_context,
        predict_frames,
        score_graph_nodes,
        score_predictions,
        sha256_file,
    )
except ModuleNotFoundError:
    from research.spotiflow_biohub.evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        EXPECTED_PARAMETER_COUNT,
        LOW_PROBABILITY_THRESHOLD,
        MATCH_RADIUS_UM,
        SPATIAL_DOWNSAMPLE,
        density_threshold,
        graph_points_by_frame,
        load_image_context,
        predict_frames,
        score_graph_nodes,
        score_predictions,
        sha256_file,
    )

try:
    from density_calibration import read_estimated_node_count, uniform_frame_indices
except ModuleNotFoundError:
    from research.density_calibration import read_estimated_node_count, uniform_frame_indices


def validate_training_result(model_dir: Path, result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or not result.get("weights_changed"):
        raise ValueError("fine-tune result is not a completed learned checkpoint")
    best = model_dir / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    actual = sha256_file(best)
    if actual != result.get("best_weight_sha256"):
        raise ValueError(f"fine-tuned checkpoint hash mismatch: {actual}")
    return result


def main(args: argparse.Namespace) -> None:
    import torch
    from spotiflow.model import Spotiflow

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fine-tuned detector acceptance")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_result = validate_training_result(args.model_dir, args.training_result)
    model = Spotiflow.from_folder(
        str(args.model_dir), which="best", inference_mode=True, map_location="cuda"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected fine-tuned parameter count: {parameter_count}")

    rows = []
    for stem in ACCEPTANCE_STEMS:
        sample_path = args.competition_dir / "train" / f"{stem}.zarr"
        truth_path = args.competition_dir / "train" / f"{stem}.geff"
        baseline_path = args.baseline_predictions / f"{stem}.geff"
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"Missing estimated count for {stem}")
        dataset, _, _, _ = load_image_context(sample_path)
        n_frames = int(dataset.image_shape[0])
        predictions, _ = predict_frames(
            model, sample_path, range(n_frames), "spotiflow_auto"
        )
        sampled_indices = set(
            uniform_frame_indices(n_frames, args.calibration_frames).tolist()
        )
        sampled_predictions = [row for row in predictions if row.frame in sampled_indices]
        threshold, projected = density_threshold(sampled_predictions, estimated, n_frames)

        # Ground-truth coordinates are first read after the image/metadata-only
        # threshold has been fixed for this movie.
        gt_by_frame = graph_points_by_frame(truth_path)
        baseline_by_frame = graph_points_by_frame(baseline_path)
        score = score_predictions(predictions, gt_by_frame, threshold)
        baseline_score = score_graph_nodes(baseline_by_frame, gt_by_frame, n_frames)
        row = {
            "stem": stem,
            "estimated_node_count": estimated,
            "threshold": threshold,
            "calibration_projected_node_count": projected,
            "calibration_projected_count_ratio": projected / estimated,
            "candidate": score,
            "baseline_raw_graph": baseline_score,
            "annotated_recall_delta": float(score["annotated_node_recall"])
            - float(baseline_score["annotated_node_recall"]),
            "final_count_ratio": float(score["predicted_nodes"]) / estimated,
            "baseline_count_ratio": float(baseline_score["predicted_nodes"]) / estimated,
        }
        rows.append(row)
        print("ACCEPTANCE", json.dumps(row, sort_keys=True), flush=True)
        (args.output_dir / "acceptance_partial.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("wall-clock guard reached during fine-tuned acceptance")

    annotated = sum(int(row["candidate"]["annotated_gt_nodes"]) for row in rows)
    matched = sum(int(row["candidate"]["matched_gt_nodes"]) for row in rows)
    baseline_matched = sum(
        int(row["baseline_raw_graph"]["matched_gt_nodes"]) for row in rows
    )
    pretrained_recall = float(args.pretrained_acceptance_recall)
    recall = matched / annotated
    summary = {
        "status": "completed",
        "elapsed_seconds": time.monotonic() - started,
        "normalization": "spotiflow_auto_fixed_before_acceptance",
        "acceptance": {
            "annotated_gt_nodes": annotated,
            "candidate_matched_gt_nodes": matched,
            "baseline_matched_gt_nodes": baseline_matched,
            "candidate_annotated_node_recall": recall,
            "baseline_annotated_node_recall": baseline_matched / annotated,
            "delta_vs_baseline": (matched - baseline_matched) / annotated,
            "delta_vs_pretrained_spotiflow": recall - pretrained_recall,
            "rows": rows,
        },
        "provenance": {
            "best_sha256": sha256_file(args.model_dir / "best.pt"),
            "training_result_sha256": sha256_file(args.training_result),
            "base_weight_sha256": training_result["base_weight_sha256"],
            "parameter_count": parameter_count,
        },
        "acceptance_stems": list(ACCEPTANCE_STEMS),
        "parameters": {
            "calibration_frames": args.calibration_frames,
            "low_probability_threshold": LOW_PROBABILITY_THRESHOLD,
            "match_radius_um": MATCH_RADIUS_UM,
            "spatial_downsample": SPATIAL_DOWNSAMPLE.tolist(),
        },
    }
    result_path = args.output_dir / "finetuned_detector_acceptance.json"
    result_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TERMINAL", json.dumps(summary["acceptance"], sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--pretrained-acceptance-recall", type=float, default=0.7776834959694527)
    parser.add_argument("--max-wall-seconds", type=float, default=6300.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
