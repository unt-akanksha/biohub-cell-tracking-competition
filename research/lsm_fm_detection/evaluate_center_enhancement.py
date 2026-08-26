#!/usr/bin/env python
"""Selection-gated evaluation of a learned feature-36 center enhancer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from center_enhancement import (
        apply_bounded_offsets,
        build_center_enhancement_model,
        extract_center_patches,
    )
    from density_calibration import read_estimated_node_count, uniform_frame_indices
    from evaluate_localization_refinement import (
        PROMOTION_WORST_DELTA_MIN,
        PUBLIC_ACCEPTANCE_RECALL,
        PUBLIC_PREFIX_RECALL,
        select_global_strategy,
        summarize_rows,
    )
    from evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        SCREEN_STEMS,
        FramePeaks,
        density_threshold,
        graph_points_by_frame,
        score_graph_nodes,
        score_predictions,
    )
    from inference import predict_probability_batch
    from localization_refinement import refine_peaks_weighted
    from pu_targets import extract_local_peaks
    from train_center_enhancement import (
        EXPECTED_DETECTOR_PARAMETERS,
        LOW_PROBABILITY_THRESHOLD,
        PATCH_SHAPE,
        CONTROL_PROBABILITY_POWER,
        CONTROL_REFINEMENT_RADIUS,
        sha256_file,
    )
    from train_spatialdino_pu_detector import normalize_spatialdino_frame
except ModuleNotFoundError:
    from research.density_calibration import (
        read_estimated_node_count,
        uniform_frame_indices,
    )
    from research.lsm_fm_detection.center_enhancement import (
        apply_bounded_offsets,
        build_center_enhancement_model,
        extract_center_patches,
    )
    from research.lsm_fm_detection.evaluate_localization_refinement import (
        PROMOTION_WORST_DELTA_MIN,
        PUBLIC_ACCEPTANCE_RECALL,
        PUBLIC_PREFIX_RECALL,
        select_global_strategy,
        summarize_rows,
    )
    from research.lsm_fm_detection.train_center_enhancement import (
        EXPECTED_DETECTOR_PARAMETERS,
        LOW_PROBABILITY_THRESHOLD,
        PATCH_SHAPE,
        CONTROL_PROBABILITY_POWER,
        CONTROL_REFINEMENT_RADIUS,
        sha256_file,
    )
    from research.spatialdino_detection.inference import predict_probability_batch
    from research.spatialdino_detection.train_pu_detector import (
        normalize_spatialdino_frame,
    )
    from research.spotiflow_biohub.evaluate_pretrained_detector import (
        ACCEPTANCE_STEMS,
        SCREEN_STEMS,
        FramePeaks,
        density_threshold,
        graph_points_by_frame,
        score_graph_nodes,
        score_predictions,
    )
    from research.spotiflow_biohub.pu_targets import extract_local_peaks
    from research.lsm_fm_detection.localization_refinement import refine_peaks_weighted


CONTROL_CANDIDATE = "feature36_control"
CANDIDATE_SCALES = {
    CONTROL_CANDIDATE: 0.0,
    "center_enhancement_half": 0.5,
    "center_enhancement_full": 1.0,
}
MAXIMUM_APPLIED_OFFSET_VOXELS = 2.0


def validate_enhancement_training(checkpoint: Path, result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("validation_overlap") != []:
        raise ValueError("center-enhancement training evidence is invalid")
    if result.get("selection_labels_read_during_training") is not False:
        raise ValueError("selection labels entered center-enhancement training")
    if result.get("acceptance_labels_read_during_training") is not False:
        raise ValueError("acceptance labels entered center-enhancement training")
    if result.get("public_predictions_copied") is not False:
        raise ValueError("training evidence does not prove independent output")
    if sha256_file(checkpoint) != result.get("checkpoint_sha256"):
        raise ValueError("center-enhancement checkpoint hash mismatch")
    targets = result.get("targets", {})
    if targets.get("unmatched_peaks_used_as_negatives") is not False:
        raise ValueError("sparse-label unknown peaks were used as negatives")
    return result


def predict_offsets(refiner, patches: np.ndarray, *, batch_size: int, device) -> np.ndarray:
    import torch

    if batch_size <= 0:
        raise ValueError("refiner batch size must be positive")
    if not len(patches):
        return np.empty((0, 3), dtype=np.float32)
    results = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start : start + batch_size]).to(device)
            results.append(refiner(batch)["offsets"].float().cpu().numpy())
    return np.concatenate(results).astype(np.float32, copy=False)


def scaled_refinements(
    centers: np.ndarray,
    offsets: np.ndarray,
    *,
    volume_shape: Sequence[int],
    candidate_scales: dict[str, float] = CANDIDATE_SCALES,
) -> dict[str, np.ndarray]:
    """Return predeclared global scales while preserving peak identities."""

    points = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    delta = np.asarray(offsets, dtype=np.float32).reshape(-1, 3)
    if points.shape != delta.shape:
        raise ValueError("centers and offsets must align")
    if not candidate_scales or CONTROL_CANDIDATE not in candidate_scales:
        raise ValueError("candidate scales must include the control")
    if candidate_scales[CONTROL_CANDIDATE] != 0.0:
        raise ValueError("control scale must be zero")
    if any(not 0.0 <= float(scale) <= 1.0 for scale in candidate_scales.values()):
        raise ValueError("candidate scales must lie in [0, 1]")
    return {
        name: apply_bounded_offsets(
            points,
            delta * float(scale),
            volume_shape=volume_shape,
            maximum_offset_voxels=MAXIMUM_APPLIED_OFFSET_VOXELS,
        )
        for name, scale in candidate_scales.items()
    }


def predict_candidates(
    detector,
    refiner,
    sample_path: Path,
    frames: Sequence[int],
    *,
    device,
    detector_batch_size: int,
    refiner_batch_size: int,
) -> dict[str, list[FramePeaks]]:
    import torch
    import zarr

    results = {name: [] for name in CANDIDATE_SCALES}
    array = zarr.open_group(str(sample_path), mode="r")["0"]
    frame_indices = [int(frame) for frame in frames]
    for start in range(0, len(frame_indices), detector_batch_size):
        batch_frames = frame_indices[start : start + detector_batch_size]
        loaded = [
            normalize_spatialdino_frame(array[frame, :, ::4, ::4].astype(np.float32))
            for frame in batch_frames
        ]
        images = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
        probabilities = predict_probability_batch(
            detector, images, yx_tta=True
        ).cpu().numpy()
        for local_index, frame in enumerate(batch_frames):
            probability = probabilities[local_index, 0]
            peaks = extract_local_peaks(
                probability,
                threshold=LOW_PROBABILITY_THRESHOLD,
                min_distance_voxels=1,
            )
            control_points = refine_peaks_weighted(
                probability,
                peaks.coords,
                radius=CONTROL_REFINEMENT_RADIUS,
                probability_power=CONTROL_PROBABILITY_POWER,
            )
            patches = extract_center_patches(
                loaded[local_index],
                probability,
                control_points,
                patch_shape=PATCH_SHAPE,
            )
            offsets = predict_offsets(
                refiner, patches, batch_size=refiner_batch_size, device=device
            )
            refinements = scaled_refinements(
                control_points, offsets, volume_shape=probability.shape
            )
            for name, points in refinements.items():
                results[name].append(
                    FramePeaks(
                        frame=frame,
                        points_input=points,
                        probabilities=peaks.confidence,
                    )
                )
        del images, probabilities
    return results


def evaluate_movies(
    detector,
    refiner,
    competition_dir: Path,
    stems: Sequence[str],
    *,
    device,
    detector_batch_size: int,
    refiner_batch_size: int,
    calibration_frames: int,
    baseline_predictions: Path | None,
    candidate_names: Sequence[str],
    partial_path: Path,
    started: float,
    max_wall_seconds: float,
) -> dict[str, list[dict[str, Any]]]:
    import zarr

    rows = {name: [] for name in candidate_names}
    for stem in stems:
        sample_path = competition_dir / "train" / f"{stem}.zarr"
        truth_path = competition_dir / "train" / f"{stem}.geff"
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"missing estimated node count for {stem}")
        frame_count = int(zarr.open_group(str(sample_path), mode="r")["0"].shape[0])
        predictions = predict_candidates(
            detector,
            refiner,
            sample_path,
            list(range(frame_count)),
            device=device,
            detector_batch_size=detector_batch_size,
            refiner_batch_size=refiner_batch_size,
        )
        sampled = set(uniform_frame_indices(frame_count, calibration_frames).tolist())
        threshold, projected = density_threshold(
            [row for row in predictions[CONTROL_CANDIDATE] if row.frame in sampled],
            estimated,
            frame_count,
        )
        truth = graph_points_by_frame(truth_path)
        baseline = (
            score_graph_nodes(
                graph_points_by_frame(baseline_predictions / f"{stem}.geff"),
                truth,
                frame_count,
            )
            if baseline_predictions is not None
            else None
        )
        for name in candidate_names:
            score = score_predictions(predictions[name], truth, threshold)
            row: dict[str, Any] = {
                "stem": stem,
                "estimated_node_count": estimated,
                "threshold": threshold,
                "projected_node_count": projected,
                "projected_count_ratio": projected / estimated,
                "candidate": score,
            }
            if baseline is not None:
                row["baseline_raw_graph"] = baseline
                row["annotated_recall_delta"] = float(score["annotated_node_recall"]) - float(
                    baseline["annotated_node_recall"]
                )
            rows[name].append(row)
        partial_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "CENTER ENHANCEMENT EVAL",
            json.dumps(
                {"stem": stem, "candidates": {name: rows[name][-1]["candidate"] for name in candidate_names}},
                sort_keys=True,
            ),
            flush=True,
        )
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("center-enhancement evaluation reached wall guard")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint-sha256", required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-training-result", type=Path, required=True)
    parser.add_argument("--enhancement-model", type=Path, required=True)
    parser.add_argument("--enhancement-training-result", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector-batch-size", type=int, default=1)
    parser.add_argument("--refiner-batch-size", type=int, default=512)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=3000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.detector_batch_size, args.refiner_batch_size, args.calibration_frames) <= 0:
        raise ValueError("batch and calibration sizes must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("center-enhancement evaluation requires CUDA")
    started = time.monotonic()
    device = torch.device("cuda")
    training = validate_enhancement_training(
        args.enhancement_model, args.enhancement_training_result
    )

    try:
        from lsm_fm_image_text_model import build_lsm_fm_detector
    except ModuleNotFoundError:
        from research.lsm_fm_detection.image_text_model import build_lsm_fm_detector

    detector_result = json.loads(args.detector_training_result.read_text(encoding="utf-8"))
    if detector_result.get("status") != "completed" or detector_result.get("validation_overlap") != []:
        raise ValueError("detector training evidence is invalid")
    if sha256_file(args.detector_model) != detector_result.get("best_weight_sha256"):
        raise ValueError("detector checkpoint hash mismatch")
    detector = build_lsm_fm_detector(
        args.lsm_fm_checkpoint,
        expected_sha256=args.lsm_fm_checkpoint_sha256,
    )
    detector.load_state_dict(
        torch.load(args.detector_model, map_location="cpu", weights_only=True)["state_dict"],
        strict=True,
    )
    if sum(parameter.numel() for parameter in detector.parameters()) != EXPECTED_DETECTOR_PARAMETERS:
        raise RuntimeError("unexpected detector parameter count")
    detector.requires_grad_(False).eval().to(device)

    checkpoint = torch.load(args.enhancement_model, map_location="cpu", weights_only=True)
    refiner = build_center_enhancement_model(channels=int(checkpoint["channels"]))
    refiner.load_state_dict(checkpoint["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in refiner.parameters()) != training["parameter_count"]:
        raise RuntimeError("unexpected center-enhancement parameter count")
    refiner.requires_grad_(False).eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = evaluate_movies(
        detector,
        refiner,
        args.competition_dir,
        SCREEN_STEMS,
        device=device,
        detector_batch_size=args.detector_batch_size,
        refiner_batch_size=args.refiner_batch_size,
        calibration_frames=args.calibration_frames,
        baseline_predictions=None,
        candidate_names=tuple(CANDIDATE_SCALES),
        partial_path=args.output_dir / "selection_center_enhancement_partial.json",
        started=started,
        max_wall_seconds=args.max_wall_seconds,
    )
    selection = {name: summarize_rows(rows) for name, rows in selection_rows.items()}
    selected, diagnostics = select_global_strategy(
        selection, control_name=CONTROL_CANDIDATE
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "selection": selection,
        "selection_diagnostics": diagnostics,
        "selected_candidate": selected,
        "selection_passed": selected is not None,
        "acceptance": None,
        "acceptance_opened": False,
        "promotion_passed": False,
        "candidate_scales": CANDIDATE_SCALES,
        "public_leaderboard_used_for_selection": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
        "provenance": {
            "detector_checkpoint_sha256": sha256_file(args.detector_model),
            "enhancement_checkpoint_sha256": sha256_file(args.enhancement_model),
            "enhancement_training_result_sha256": sha256_file(args.enhancement_training_result),
            "detector_parameters": EXPECTED_DETECTOR_PARAMETERS,
            "enhancement_parameters": training["parameter_count"],
            "count_preserving": True,
            "confidence_preserving": True,
            "maximum_applied_offset_voxels": MAXIMUM_APPLIED_OFFSET_VOXELS,
            "residual_control": "probability_r2_p2",
        },
    }
    if selected is not None:
        acceptance_rows = evaluate_movies(
            detector,
            refiner,
            args.competition_dir,
            ACCEPTANCE_STEMS,
            device=device,
            detector_batch_size=args.detector_batch_size,
            refiner_batch_size=args.refiner_batch_size,
            calibration_frames=args.calibration_frames,
            baseline_predictions=args.baseline_predictions,
            candidate_names=(selected,),
            partial_path=args.output_dir / "acceptance_center_enhancement_partial.json",
            started=started,
            max_wall_seconds=args.max_wall_seconds,
        )[selected]
        acceptance = summarize_rows(acceptance_rows)
        deltas = [float(row["annotated_recall_delta"]) for row in acceptance_rows]
        prefix_pass = all(
            acceptance["by_prefix"][prefix]["annotated_node_recall"]
            >= PUBLIC_PREFIX_RECALL[prefix] - 0.01
            for prefix in PUBLIC_PREFIX_RECALL
        )
        result.update(
            {
                "acceptance": acceptance,
                "acceptance_opened": True,
                "promotion_passed": (
                    acceptance["annotated_node_recall"] >= PUBLIC_ACCEPTANCE_RECALL
                    and min(deltas) >= PROMOTION_WORST_DELTA_MIN
                    and prefix_pass
                ),
            }
        )
    output = args.output_dir / "lsm_fm_center_enhancement.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CENTER ENHANCEMENT VALIDATION COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
