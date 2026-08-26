#!/usr/bin/env python
"""Clean selection and untouched acceptance for the SpatialDINO PU detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

try:
    from density_calibration import read_estimated_node_count, uniform_frame_indices
    from encoder import load_spatialdino_vits8, sha256_file
    from evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        SCREEN_STEMS,
        density_threshold,
        graph_points_by_frame,
        score_graph_nodes,
        score_predictions,
    )
    from inference import predict_frames
    from model import HybridSpatialDinoDetector
    from train_spatialdino_pu_detector import (
        EXPECTED_MODEL_PARAMETERS,
        SPATIALDINO_SHA256,
    )
except ModuleNotFoundError:
    from research.density_calibration import read_estimated_node_count, uniform_frame_indices
    from research.spatialdino_association.encoder import (
        load_spatialdino_vits8,
        sha256_file,
    )
    from research.spatialdino_detection.inference import predict_frames
    from research.spatialdino_detection.model import HybridSpatialDinoDetector
    from research.spatialdino_detection.train_pu_detector import (
        EXPECTED_MODEL_PARAMETERS,
        SPATIALDINO_SHA256,
    )
    from research.spotiflow_biohub.evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        SCREEN_STEMS,
        density_threshold,
        graph_points_by_frame,
        score_graph_nodes,
        score_predictions,
    )


PUBLIC_ACCEPTANCE_RECALL = 0.96902842596521
PUBLIC_PREFIX_RECALL = {"44b6": 0.9588014981273408, "6bba": 0.9775019394879751}
SELECTION_POOLED_MIN = 0.80
SELECTION_WORST_MOVIE_MIN = 0.65
PROMOTION_WORST_DELTA_MIN = -0.01


def validate_training_result(model_path: Path, result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or not result.get("ema_checkpoint"):
        raise ValueError("SpatialDINO PU training result is not a completed EMA checkpoint")
    if result.get("validation_overlap") != [] or result.get("public_predictions_copied") is not False:
        raise ValueError("training provenance violated clean independent learning")
    if sha256_file(model_path) != result.get("best_weight_sha256"):
        raise ValueError("SpatialDINO PU checkpoint hash does not match training result")
    return result


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    annotated = sum(int(row["candidate"]["annotated_gt_nodes"]) for row in rows)
    matched = sum(int(row["candidate"]["matched_gt_nodes"]) for row in rows)
    by_prefix = {}
    for prefix in ("44b6", "6bba"):
        prefix_rows = [row for row in rows if str(row["stem"]).startswith(prefix)]
        prefix_annotated = sum(int(row["candidate"]["annotated_gt_nodes"]) for row in prefix_rows)
        prefix_matched = sum(int(row["candidate"]["matched_gt_nodes"]) for row in prefix_rows)
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
        "worst_movie_recall": min(float(row["candidate"]["annotated_node_recall"]) for row in rows),
        "by_prefix": by_prefix,
        "rows": list(rows),
    }


def evaluate_stems(
    model,
    competition_dir: Path,
    stems: Sequence[str],
    *,
    device,
    batch_size: int,
    calibration_frames: int,
    baseline_predictions: Path | None,
    max_wall_seconds: float,
    started: float,
    partial_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for stem in stems:
        sample_path = competition_dir / "train" / f"{stem}.zarr"
        truth_path = competition_dir / "train" / f"{stem}.geff"
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"missing estimated node count for {stem}")
        import zarr

        frame_count = int(zarr.open_group(str(sample_path), mode="r")["0"].shape[0])
        predictions, _ = predict_frames(
            model,
            sample_path,
            range(frame_count),
            device=device,
            batch_size=batch_size,
            yx_tta=True,
        )
        sampled = set(uniform_frame_indices(frame_count, calibration_frames).tolist())
        threshold, projected = density_threshold(
            [row for row in predictions if row.frame in sampled], estimated, frame_count
        )
        # Ground truth is first read after the image/metadata threshold is frozen.
        truth = graph_points_by_frame(truth_path)
        candidate = score_predictions(predictions, truth, threshold)
        row: dict[str, Any] = {
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
        partial_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print("DINO PU EVAL", json.dumps(row, sort_keys=True), flush=True)
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("SpatialDINO PU evaluation reached its wall guard")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--spatialdino-checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=1300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("batch size and wall budget must be positive")
    started = time.monotonic()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SpatialDINO PU evaluation requires CUDA")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_result = validate_training_result(args.model_path, args.training_result)
    encoder = load_spatialdino_vits8(
        args.spatialdino_checkpoint,
        expected_sha256=SPATIALDINO_SHA256,
        map_location="cpu",
    )
    model = HybridSpatialDinoDetector(encoder)
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError("unexpected hybrid detector parameter count")
    model.requires_grad_(False).eval().to(device)

    selection_rows = evaluate_stems(
        model,
        args.competition_dir,
        SCREEN_STEMS,
        device=device,
        batch_size=args.batch_size,
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
    result: dict[str, Any] = {
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
        "public_predictions_copied": False,
        "competition_submission_performed": False,
        "provenance": {
            "checkpoint_sha256": sha256_file(args.model_path),
            "training_result_sha256": sha256_file(args.training_result),
            "parameter_count": EXPECTED_MODEL_PARAMETERS,
            "actual_training_steps": training_result["actual_steps"],
        },
    }
    if selection_passed:
        acceptance_rows = evaluate_stems(
            model,
            args.competition_dir,
            ACCEPTANCE_STEMS,
            device=device,
            batch_size=args.batch_size,
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
    output = args.output_dir / "spatialdino_pu_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DINO PU VALIDATION COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
