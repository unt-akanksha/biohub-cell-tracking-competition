#!/usr/bin/env python
"""Clean selection of two independently pretrained LSM-FM detector variants."""

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
    from evaluate_localization_refinement import (
        PUBLIC_ACCEPTANCE_RECALL,
        PUBLIC_PREFIX_RECALL,
        SELECTION_MOVIE_REGRESSION_MIN,
        SELECTION_POOLED_MIN,
        SELECTION_WORST_MOVIE_MIN,
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
    from localization_refinement import refine_peaks_log_quadratic, refine_peaks_weighted
    from pu_targets import extract_local_peaks
    from train_spatialdino_pu_detector import normalize_spatialdino_frame
except ModuleNotFoundError:
    from research.density_calibration import read_estimated_node_count, uniform_frame_indices
    from research.lsm_fm_detection.evaluate_localization_refinement import (
        PUBLIC_ACCEPTANCE_RECALL,
        PUBLIC_PREFIX_RECALL,
        SELECTION_MOVIE_REGRESSION_MIN,
        SELECTION_POOLED_MIN,
        SELECTION_WORST_MOVIE_MIN,
        summarize_rows,
    )
    from research.lsm_fm_detection.localization_refinement import (
        refine_peaks_log_quadratic,
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


LOW_PROBABILITY_THRESHOLD = 0.02
REFERENCE_CANDIDATE = "feature36_control"
PROMOTION_WORST_DELTA_MIN = -0.01


@dataclass(frozen=True)
class Candidate:
    name: str
    feature24_weight: float
    feature36_weight: float
    refinement: str = "centroid"


CANDIDATES = (
    Candidate("feature24_control", 1.0, 0.0),
    Candidate(REFERENCE_CANDIDATE, 0.0, 1.0),
    Candidate("feature36_log_quadratic", 0.0, 1.0, "log_quadratic"),
    Candidate("equal_ensemble_control", 0.5, 0.5),
    Candidate("equal_ensemble_log_quadratic", 0.5, 0.5, "log_quadratic"),
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_probabilities(
    feature24: np.ndarray,
    feature36: np.ndarray,
    candidate: Candidate,
) -> np.ndarray:
    if candidate.feature24_weight < 0 or candidate.feature36_weight < 0:
        raise ValueError("ensemble weights must be nonnegative")
    total = candidate.feature24_weight + candidate.feature36_weight
    if total <= 0:
        raise ValueError("at least one ensemble weight must be positive")
    first = np.asarray(feature24, dtype=np.float32)
    second = np.asarray(feature36, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("model probabilities must be equal-shape 3D volumes")
    return (
        first * candidate.feature24_weight + second * candidate.feature36_weight
    ) / total


def select_global_candidate(
    summaries: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    reference_rows = {
        str(row["stem"]): float(row["candidate"]["annotated_node_recall"])
        for row in summaries[REFERENCE_CANDIDATE]["rows"]
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    eligible = []
    for name, summary in summaries.items():
        deltas = [
            float(row["candidate"]["annotated_node_recall"])
            - reference_rows[str(row["stem"])]
            for row in summary["rows"]
        ]
        passed = (
            float(summary["annotated_node_recall"]) >= SELECTION_POOLED_MIN
            and float(summary["worst_movie_recall"]) >= SELECTION_WORST_MOVIE_MIN
            and min(deltas) >= SELECTION_MOVIE_REGRESSION_MIN
        )
        diagnostics[name] = {
            "selection_passed": passed,
            "minimum_movie_delta_from_feature36": min(deltas),
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
            name == REFERENCE_CANDIDATE,
        ),
    )
    return selected, diagnostics


def predict_candidates(
    models: dict[str, Any],
    sample_path: Path,
    frames: Sequence[int],
    *,
    device: Any,
    batch_size: int,
    candidates: Sequence[Candidate],
) -> dict[str, list[FramePeaks]]:
    import torch
    import zarr

    array = zarr.open_group(str(sample_path), mode="r")["0"]
    frame_indices = [int(frame) for frame in frames]
    results = {candidate.name: [] for candidate in candidates}
    for start in range(0, len(frame_indices), batch_size):
        batch_frames = frame_indices[start : start + batch_size]
        loaded = [
            normalize_spatialdino_frame(array[frame, :, ::4, ::4].astype(np.float32))
            for frame in batch_frames
        ]
        images = torch.from_numpy(np.stack(loaded)[:, None]).to(device)
        probabilities = {
            name: predict_probability_batch(model, images, yx_tta=True).cpu().numpy()
            for name, model in models.items()
        }
        for batch_index, frame in enumerate(batch_frames):
            feature24 = probabilities["feature24"][batch_index, 0]
            feature36 = probabilities["feature36"][batch_index, 0]
            for candidate in candidates:
                probability = combine_probabilities(feature24, feature36, candidate)
                peak_set = extract_local_peaks(
                    probability,
                    threshold=LOW_PROBABILITY_THRESHOLD,
                    min_distance_voxels=1,
                )
                if candidate.refinement == "log_quadratic":
                    points = refine_peaks_log_quadratic(probability, peak_set.coords)
                elif candidate.refinement == "centroid":
                    points = refine_peaks_weighted(
                        probability,
                        peak_set.coords,
                        radius=1,
                        probability_power=2.0,
                    )
                else:
                    raise ValueError(f"unknown refinement: {candidate.refinement}")
                results[candidate.name].append(
                    FramePeaks(
                        frame=frame,
                        points_input=points,
                        probabilities=peak_set.confidence,
                    )
                )
        del images, probabilities
    return results


def evaluate_movies(
    models: dict[str, Any],
    competition_dir: Path,
    stems: Sequence[str],
    *,
    device: Any,
    batch_size: int,
    calibration_frames: int,
    candidates: Sequence[Candidate],
    baseline_predictions: Path | None,
    max_wall_seconds: float,
    started: float,
    partial_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    import zarr

    rows = {candidate.name: [] for candidate in candidates}
    for stem in stems:
        sample_path = competition_dir / "train" / f"{stem}.zarr"
        truth_path = competition_dir / "train" / f"{stem}.geff"
        estimated = read_estimated_node_count(truth_path)
        if estimated is None:
            raise ValueError(f"missing estimated node count for {stem}")
        frame_count = int(zarr.open_group(str(sample_path), mode="r")["0"].shape[0])
        predictions = predict_candidates(
            models,
            sample_path,
            list(range(frame_count)),
            device=device,
            batch_size=batch_size,
            candidates=candidates,
        )
        sampled = set(uniform_frame_indices(frame_count, calibration_frames).tolist())
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
        for candidate in candidates:
            values = predictions[candidate.name]
            threshold, projected = density_threshold(
                [row for row in values if row.frame in sampled], estimated, frame_count
            )
            score = score_predictions(values, truth, threshold)
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
            rows[candidate.name].append(row)
        partial_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "LSM-FM ENSEMBLE EVAL",
            json.dumps(
                {stem: {name: value[-1]["candidate"] for name, value in rows.items()}},
                sort_keys=True,
            ),
            flush=True,
        )
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("LSM-FM ensemble evaluation reached its wall guard")
    return rows


def validated_training_result(model_path: Path, result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError("training result is not complete")
    if result.get("validation_overlap") != []:
        raise ValueError("training result has validation overlap")
    if result.get("public_predictions_copied") is not False:
        raise ValueError("training result does not prove independent output")
    if sha256_file(model_path) != result.get("best_weight_sha256"):
        raise ValueError("learned checkpoint hash mismatch")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--feature24-base", type=Path, required=True)
    parser.add_argument("--feature24-base-sha256", required=True)
    parser.add_argument("--feature24-model", type=Path, required=True)
    parser.add_argument("--feature24-training-result", type=Path, required=True)
    parser.add_argument("--feature36-base", type=Path, required=True)
    parser.add_argument("--feature36-base-sha256", required=True)
    parser.add_argument("--feature36-model", type=Path, required=True)
    parser.add_argument("--feature36-training-result", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("batch size and wall budget must be positive")
    started = time.monotonic()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("LSM-FM ensemble evaluation requires CUDA")
    validated_training_result(args.feature24_model, args.feature24_training_result)
    validated_training_result(args.feature36_model, args.feature36_training_result)
    try:
        from lsm_fm_image_only_model import (
            EXPECTED_DETECTOR_PARAMETERS as feature24_parameters,
            build_lsm_fm_detector as build_feature24,
        )
        from lsm_fm_image_text_model import (
            EXPECTED_DETECTOR_PARAMETERS as feature36_parameters,
            build_lsm_fm_detector as build_feature36,
        )
    except ModuleNotFoundError:
        from research.lsm_fm_detection.image_text_model import (
            EXPECTED_DETECTOR_PARAMETERS as feature36_parameters,
            build_lsm_fm_detector as build_feature36,
        )
        from research.lsm_fm_detection.model import (
            EXPECTED_DETECTOR_PARAMETERS as feature24_parameters,
            build_lsm_fm_detector as build_feature24,
        )

    models = {
        "feature24": build_feature24(
            args.feature24_base, expected_sha256=args.feature24_base_sha256
        ),
        "feature36": build_feature36(
            args.feature36_base, expected_sha256=args.feature36_base_sha256
        ),
    }
    for name, model_path, expected_parameters in (
        ("feature24", args.feature24_model, feature24_parameters),
        ("feature36", args.feature36_model, feature36_parameters),
    ):
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        models[name].load_state_dict(payload["state_dict"], strict=True)
        if sum(parameter.numel() for parameter in models[name].parameters()) != expected_parameters:
            raise RuntimeError(f"unexpected {name} detector parameter count")
    device = torch.device("cuda")
    for model in models.values():
        model.requires_grad_(False).eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows = evaluate_movies(
        models,
        args.competition_dir,
        SCREEN_STEMS,
        device=device,
        batch_size=args.batch_size,
        calibration_frames=args.calibration_frames,
        candidates=CANDIDATES,
        baseline_predictions=None,
        max_wall_seconds=args.max_wall_seconds,
        started=started,
        partial_path=args.output_dir / "selection_ensemble_partial.json",
    )
    selection = {name: summarize_rows(rows) for name, rows in selection_rows.items()}
    selected, diagnostics = select_global_candidate(selection)
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
        "public_leaderboard_used_for_selection": False,
        "public_predictions_copied": False,
        "competition_submission_performed": False,
        "provenance": {
            "feature24_checkpoint_sha256": sha256_file(args.feature24_model),
            "feature24_training_result_sha256": sha256_file(args.feature24_training_result),
            "feature36_checkpoint_sha256": sha256_file(args.feature36_model),
            "feature36_training_result_sha256": sha256_file(args.feature36_training_result),
            "feature24_parameters": feature24_parameters,
            "feature36_parameters": feature36_parameters,
            "candidate_scope": "two single models and a fixed equal-probability ensemble with global coordinate refinement",
        },
    }
    if selected is not None:
        candidate = next(value for value in CANDIDATES if value.name == selected)
        acceptance_rows = evaluate_movies(
            models,
            args.competition_dir,
            ACCEPTANCE_STEMS,
            device=device,
            batch_size=args.batch_size,
            calibration_frames=args.calibration_frames,
            candidates=(candidate,),
            baseline_predictions=args.baseline_predictions,
            max_wall_seconds=args.max_wall_seconds,
            started=started,
            partial_path=args.output_dir / "acceptance_ensemble_partial.json",
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
    output = args.output_dir / "lsm_fm_ensemble_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LSM-FM ENSEMBLE COMPLETE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
