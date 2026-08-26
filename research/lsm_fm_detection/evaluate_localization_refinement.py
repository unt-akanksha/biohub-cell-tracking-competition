#!/usr/bin/env python
"""Selection-gated coordinate refinement for a frozen LSM-FM checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from density_calibration import read_estimated_node_count, uniform_frame_indices
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
    from localization_refinement import (
        refine_peaks_log_quadratic,
        refine_peaks_quadratic,
        refine_peaks_weighted,
    )
    from pu_targets import extract_local_peaks
    from train_spatialdino_pu_detector import normalize_spatialdino_frame
except ModuleNotFoundError:
    from research.density_calibration import read_estimated_node_count, uniform_frame_indices
    from research.lsm_fm_detection.localization_refinement import (
        refine_peaks_log_quadratic,
        refine_peaks_quadratic,
        refine_peaks_weighted,
    )
    from research.spatialdino_detection.inference import predict_probability_batch
    from research.spatialdino_detection.train_pu_detector import normalize_spatialdino_frame
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


PUBLIC_ACCEPTANCE_RECALL = 0.96902842596521
PUBLIC_PREFIX_RECALL = {"44b6": 0.9588014981273408, "6bba": 0.9775019394879751}
SELECTION_POOLED_MIN = 0.80
SELECTION_WORST_MOVIE_MIN = 0.65
SELECTION_MOVIE_REGRESSION_MIN = -0.01
PROMOTION_WORST_DELTA_MIN = -0.01
LOW_PROBABILITY_THRESHOLD = 0.02
CONTROL_STRATEGY = "probability_r1_p2"


@dataclass(frozen=True)
class Strategy:
    name: str
    method: str
    radius: int = 1
    probability_power: float = 2.0
    intensity_power: float = 0.0


STRATEGIES = (
    Strategy(CONTROL_STRATEGY, "weighted", 1, 2.0, 0.0),
    Strategy("probability_r2_p2", "weighted", 2, 2.0, 0.0),
    Strategy("probability_r2_p4", "weighted", 2, 4.0, 0.0),
    Strategy("joint_r1_p2_i1", "weighted", 1, 2.0, 1.0),
    Strategy("joint_r2_p2_i1", "weighted", 2, 2.0, 1.0),
    Strategy("joint_r2_p4_i1", "weighted", 2, 4.0, 1.0),
    Strategy("concave_quadratic_3x3x3", "quadratic"),
    Strategy("log_probability_quadratic_3x3x3", "log_quadratic"),
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    annotated = sum(int(row["candidate"]["annotated_gt_nodes"]) for row in rows)
    matched = sum(int(row["candidate"]["matched_gt_nodes"]) for row in rows)
    by_prefix = {}
    for prefix in ("44b6", "6bba"):
        subset = [row for row in rows if str(row["stem"]).startswith(prefix)]
        prefix_annotated = sum(int(row["candidate"]["annotated_gt_nodes"]) for row in subset)
        prefix_matched = sum(int(row["candidate"]["matched_gt_nodes"]) for row in subset)
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
        "mean_movie_match_distance_um": float(
            np.mean([float(row["candidate"]["mean_match_distance_um"]) for row in rows])
        ),
        "by_prefix": by_prefix,
        "rows": list(rows),
    }


def refine(strategy: Strategy, probability: np.ndarray, intensity: np.ndarray, coords: np.ndarray) -> np.ndarray:
    if strategy.method == "quadratic":
        return refine_peaks_quadratic(probability, coords)
    if strategy.method == "log_quadratic":
        return refine_peaks_log_quadratic(probability, coords)
    return refine_peaks_weighted(
        probability,
        coords,
        intensity=intensity if strategy.intensity_power else None,
        radius=strategy.radius,
        probability_power=strategy.probability_power,
        intensity_power=strategy.intensity_power,
    )


def predict_strategies(
    model,
    sample_path: Path,
    frames: Sequence[int],
    *,
    device,
    batch_size: int,
    strategies: Sequence[Strategy],
) -> dict[str, list[FramePeaks]]:
    import torch
    import zarr

    array = zarr.open_group(str(sample_path), mode="r")["0"]
    frame_indices = [int(frame) for frame in frames]
    results = {strategy.name: [] for strategy in strategies}
    for start in range(0, len(frame_indices), batch_size):
        batch_frames = frame_indices[start : start + batch_size]
        loaded = [
            normalize_spatialdino_frame(array[frame, :, ::4, ::4].astype(np.float32))
            for frame in batch_frames
        ]
        images = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
        probabilities = predict_probability_batch(model, images, yx_tta=True).cpu().numpy()
        for batch_index, frame in enumerate(batch_frames):
            probability = probabilities[batch_index, 0]
            peak_set = extract_local_peaks(
                probability,
                threshold=LOW_PROBABILITY_THRESHOLD,
                min_distance_voxels=1,
            )
            for strategy in strategies:
                points = refine(strategy, probability, loaded[batch_index], peak_set.coords)
                results[strategy.name].append(
                    FramePeaks(
                        frame=frame,
                        points_input=points,
                        probabilities=peak_set.confidence,
                    )
                )
        del images, probabilities
    return results


def evaluate_movies(
    model,
    competition_dir: Path,
    stems: Sequence[str],
    *,
    device,
    batch_size: int,
    calibration_frames: int,
    strategies: Sequence[Strategy],
    baseline_predictions: Path | None,
    max_wall_seconds: float,
    started: float,
    partial_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    rows = {strategy.name: [] for strategy in strategies}
    for stem in stems:
        sample_path = competition_dir / "train" / f"{stem}.zarr"
        truth_path = competition_dir / "train" / f"{stem}.geff"
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"missing estimated node count for {stem}")
        import zarr

        frame_count = int(zarr.open_group(str(sample_path), mode="r")["0"].shape[0])
        predictions = predict_strategies(
            model,
            sample_path,
            list(range(frame_count)),
            device=device,
            batch_size=batch_size,
            strategies=strategies,
        )
        sampled = set(uniform_frame_indices(frame_count, calibration_frames).tolist())
        control = predictions[strategies[0].name]
        threshold, projected = density_threshold(
            [row for row in control if row.frame in sampled], estimated, frame_count
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
        for strategy in strategies:
            candidate = score_predictions(predictions[strategy.name], truth, threshold)
            row: dict[str, Any] = {
                "stem": stem,
                "estimated_node_count": estimated,
                "threshold": threshold,
                "projected_node_count": projected,
                "projected_count_ratio": projected / estimated,
                "candidate": candidate,
            }
            if baseline is not None:
                row["baseline_raw_graph"] = baseline
                row["annotated_recall_delta"] = float(candidate["annotated_node_recall"]) - float(
                    baseline["annotated_node_recall"]
                )
            rows[strategy.name].append(row)
        partial_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "LSM-FM REFINEMENT EVAL",
            json.dumps(
                {
                    "stem": stem,
                    "strategies": {
                        name: values[-1]["candidate"] for name, values in rows.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("localization-refinement evaluation reached its wall guard")
    return rows


def select_global_strategy(
    summaries: dict[str, dict[str, Any]],
    *,
    control_name: str = CONTROL_STRATEGY,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    control_rows = {
        str(row["stem"]): float(row["candidate"]["annotated_node_recall"])
        for row in summaries[control_name]["rows"]
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    eligible = []
    for name, summary in summaries.items():
        deltas = [
            float(row["candidate"]["annotated_node_recall"]) - control_rows[str(row["stem"])]
            for row in summary["rows"]
        ]
        passed = (
            float(summary["annotated_node_recall"]) >= SELECTION_POOLED_MIN
            and float(summary["worst_movie_recall"]) >= SELECTION_WORST_MOVIE_MIN
            and min(deltas) >= SELECTION_MOVIE_REGRESSION_MIN
        )
        diagnostics[name] = {
            "selection_passed": passed,
            "minimum_movie_delta_from_control": min(deltas),
        }
        if passed:
            eligible.append(name)
    if not eligible:
        return None, diagnostics
    selected = max(
        eligible,
        key=lambda name: (
            float(summaries[name]["worst_movie_recall"]),
            float(summaries[name]["annotated_node_recall"]),
            -float(summaries[name]["mean_movie_match_distance_um"]),
            name == control_name,
        ),
    )
    return selected, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint", type=Path, required=True)
    parser.add_argument("--lsm-fm-checkpoint-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=2600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("batch size and wall budget must be positive")
    started = time.monotonic()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("localization-refinement evaluation requires CUDA")
    training_result = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training_result.get("status") != "completed":
        raise ValueError("training result is not complete")
    if training_result.get("validation_overlap") != []:
        raise ValueError("training result has validation overlap")
    if training_result.get("public_predictions_copied") is not False:
        raise ValueError("training result does not prove independent output")
    if sha256_file(args.model_path) != training_result.get("best_weight_sha256"):
        raise ValueError("learned checkpoint hash mismatch")

    try:
        from lsm_fm_model import EXPECTED_DETECTOR_PARAMETERS, build_lsm_fm_detector
    except ModuleNotFoundError:
        from research.lsm_fm_detection.image_text_model import (
            EXPECTED_DETECTOR_PARAMETERS,
            build_lsm_fm_detector,
        )

    model = build_lsm_fm_detector(
        args.lsm_fm_checkpoint,
        expected_sha256=args.lsm_fm_checkpoint_sha256,
    )
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_DETECTOR_PARAMETERS:
        raise RuntimeError("unexpected detector parameter count")
    device = torch.device("cuda")
    model.requires_grad_(False).eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = evaluate_movies(
        model,
        args.competition_dir,
        SCREEN_STEMS,
        device=device,
        batch_size=args.batch_size,
        calibration_frames=args.calibration_frames,
        strategies=STRATEGIES,
        baseline_predictions=None,
        max_wall_seconds=args.max_wall_seconds,
        started=started,
        partial_path=args.output_dir / "selection_refinement_partial.json",
    )
    selection = {name: summarize_rows(rows) for name, rows in selection_rows.items()}
    selected, diagnostics = select_global_strategy(selection)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "selection": selection,
        "selection_diagnostics": diagnostics,
        "selected_strategy": selected,
        "selection_passed": selected is not None,
        "acceptance": None,
        "acceptance_opened": False,
        "promotion_passed": False,
        "public_leaderboard_used_for_selection": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
        "provenance": {
            "checkpoint_sha256": sha256_file(args.model_path),
            "training_result_sha256": sha256_file(args.training_result),
            "parameter_count": EXPECTED_DETECTOR_PARAMETERS,
            "strategy_scope": "one global coordinate refiner; confidence ranking and count calibration unchanged",
        },
    }
    if selected is not None:
        strategy = next(value for value in STRATEGIES if value.name == selected)
        acceptance_rows = evaluate_movies(
            model,
            args.competition_dir,
            ACCEPTANCE_STEMS,
            device=device,
            batch_size=args.batch_size,
            calibration_frames=args.calibration_frames,
            strategies=(strategy,),
            baseline_predictions=args.baseline_predictions,
            max_wall_seconds=args.max_wall_seconds,
            started=started,
            partial_path=args.output_dir / "acceptance_refinement_partial.json",
        )[selected]
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
            }
        )
    output = args.output_dir / "lsm_fm_localization_refinement.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LSM-FM REFINEMENT COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
