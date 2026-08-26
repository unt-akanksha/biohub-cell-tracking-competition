#!/usr/bin/env python
"""Clean selection/acceptance gate for the positive-unlabeled detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

try:
    from evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        EXPECTED_PARAMETER_COUNT,
        LOW_PROBABILITY_THRESHOLD,
        MATCH_RADIUS_UM,
        SCREEN_STEMS,
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
        SCREEN_STEMS,
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
    from research.density_calibration import (
        read_estimated_node_count,
        uniform_frame_indices,
    )


PUBLIC_ACCEPTANCE_RECALL = 0.96902842596521
PUBLIC_PREFIX_RECALL = {"44b6": 0.9588014981273408, "6bba": 0.9775019394879751}
SELECTION_POOLED_MIN = 0.80
SELECTION_WORST_MOVIE_MIN = 0.65
PROMOTION_WORST_DELTA_MIN = -0.01


def validate_training_result(model_dir: Path, result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or not result.get("weights_changed"):
        raise ValueError("PU training result is not a completed learned checkpoint")
    if result.get("validation_overlap") != []:
        raise ValueError("PU training result reports validation overlap")
    best = model_dir / "best.pt"
    if sha256_file(best) != result.get("best_weight_sha256"):
        raise ValueError("PU checkpoint hash does not match its training result")
    return result


def summarize_rows(rows: Sequence[dict]) -> dict:
    annotated = sum(int(row["candidate"]["annotated_gt_nodes"]) for row in rows)
    matched = sum(int(row["candidate"]["matched_gt_nodes"]) for row in rows)
    by_prefix = {}
    for prefix in ("44b6", "6bba"):
        prefix_rows = [row for row in rows if str(row["stem"]).startswith(prefix)]
        prefix_annotated = sum(
            int(row["candidate"]["annotated_gt_nodes"]) for row in prefix_rows
        )
        prefix_matched = sum(
            int(row["candidate"]["matched_gt_nodes"]) for row in prefix_rows
        )
        by_prefix[prefix] = {
            "annotated_gt_nodes": prefix_annotated,
            "matched_gt_nodes": prefix_matched,
            "annotated_node_recall": prefix_matched / prefix_annotated,
        }
    return {
        "movies": len(rows),
        "annotated_gt_nodes": annotated,
        "matched_gt_nodes": matched,
        "annotated_node_recall": matched / annotated,
        "worst_movie_recall": min(
            float(row["candidate"]["annotated_node_recall"]) for row in rows
        ),
        "by_prefix": by_prefix,
        "rows": list(rows),
    }


def evaluate_stems(
    model,
    competition_dir: Path,
    stems: Sequence[str],
    *,
    calibration_frames: int,
    baseline_predictions: Path | None,
    max_wall_seconds: float,
    started: float,
    partial_path: Path,
) -> list[dict]:
    rows = []
    for stem in stems:
        sample_path = competition_dir / "train" / f"{stem}.zarr"
        truth_path = competition_dir / "train" / f"{stem}.geff"
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"missing estimated node count for {stem}")
        dataset, _, _, _ = load_image_context(sample_path)
        frame_count = int(dataset.image_shape[0])
        predictions, _ = predict_frames(
            model, sample_path, range(frame_count), "spotiflow_auto"
        )
        sampled = set(
            uniform_frame_indices(frame_count, calibration_frames).tolist()
        )
        threshold, projected = density_threshold(
            [row for row in predictions if row.frame in sampled], estimated, frame_count
        )
        # Labels are opened only after the image/metadata threshold is frozen.
        truth = graph_points_by_frame(truth_path)
        candidate = score_predictions(predictions, truth, threshold)
        row = {
            "stem": stem,
            "estimated_node_count": estimated,
            "threshold": threshold,
            "projected_node_count": projected,
            "projected_count_ratio": projected / estimated,
            "candidate": candidate,
        }
        if baseline_predictions is not None:
            baseline = score_graph_nodes(
                graph_points_by_frame(baseline_predictions / f"{stem}.geff"),
                truth,
                frame_count,
            )
            row["baseline_raw_graph"] = baseline
            row["annotated_recall_delta"] = float(candidate["annotated_node_recall"]) - float(
                baseline["annotated_node_recall"]
            )
        rows.append(row)
        partial_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("PU EVAL", json.dumps(row, sort_keys=True), flush=True)
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("PU detector evaluation reached its wall guard")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=1200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    import torch
    from spotiflow.model import Spotiflow

    if not torch.cuda.is_available():
        raise RuntimeError("PU detector evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_result = validate_training_result(args.model_dir, args.training_result)
    model = Spotiflow.from_folder(
        str(args.model_dir), which="best", inference_mode=True, map_location="cuda"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected Spotiflow parameter count: {parameter_count}")

    selection_rows = evaluate_stems(
        model,
        args.competition_dir,
        SCREEN_STEMS,
        calibration_frames=args.calibration_frames,
        baseline_predictions=None,
        max_wall_seconds=args.max_wall_seconds,
        started=started,
        partial_path=args.output_dir / "selection_partial.json",
    )
    selection = summarize_rows(selection_rows)
    selection_passed = (
        selection["annotated_node_recall"] >= SELECTION_POOLED_MIN
        and selection["worst_movie_recall"] >= SELECTION_WORST_MOVIE_MIN
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "selection": selection,
        "selection_passed": selection_passed,
        "selection_gate": {
            "pooled_recall_min": SELECTION_POOLED_MIN,
            "worst_movie_recall_min": SELECTION_WORST_MOVIE_MIN,
        },
        "acceptance": None,
        "acceptance_opened": False,
        "promotion_passed": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "provenance": {
            "checkpoint_sha256": sha256_file(args.model_dir / "best.pt"),
            "training_result_sha256": sha256_file(args.training_result),
            "parameter_count": parameter_count,
            "actual_training_steps": training_result["actual_steps"],
        },
    }
    if selection_passed:
        acceptance_rows = evaluate_stems(
            model,
            args.competition_dir,
            ACCEPTANCE_STEMS,
            calibration_frames=args.calibration_frames,
            baseline_predictions=args.baseline_predictions,
            max_wall_seconds=args.max_wall_seconds,
            started=started,
            partial_path=args.output_dir / "acceptance_partial.json",
        )
        acceptance = summarize_rows(acceptance_rows)
        deltas = [float(row["annotated_recall_delta"]) for row in acceptance_rows]
        prefix_pass = all(
            acceptance["by_prefix"][prefix]["annotated_node_recall"]
            >= PUBLIC_PREFIX_RECALL[prefix] - 0.01
            for prefix in PUBLIC_PREFIX_RECALL
        )
        promotion_passed = (
            acceptance["annotated_node_recall"] >= PUBLIC_ACCEPTANCE_RECALL
            and min(deltas) >= PROMOTION_WORST_DELTA_MIN
            and prefix_pass
        )
        result.update(
            {
                "acceptance": acceptance,
                "acceptance_opened": True,
                "promotion_passed": promotion_passed,
                "promotion_gate": {
                    "public_pooled_recall": PUBLIC_ACCEPTANCE_RECALL,
                    "public_prefix_recall": PUBLIC_PREFIX_RECALL,
                    "prefix_regression_tolerance": 0.01,
                    "worst_movie_delta_min": PROMOTION_WORST_DELTA_MIN,
                },
            }
        )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    result_path = args.output_dir / "pu_detector_validation.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("PU VALIDATION TERMINAL", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
