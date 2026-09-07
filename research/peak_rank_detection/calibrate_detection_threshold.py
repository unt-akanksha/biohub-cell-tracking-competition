#!/usr/bin/env python
"""Freeze clean detector thresholds on complete synthetic selection labels.

The output is checkpoint-bound and TTA-specific.  It never reads competition
data, organizer node-count metadata, public predictions, or leaderboard scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.peak_rank_detection.evaluate_peak_rank_detector import (
    load_model,
    validate_training,
)
from research.peak_rank_detection.inference import (
    LOW_PROBABILITY_THRESHOLD,
    TTA_TRANSFORMS,
    peaks_from_prediction,
    predict_probability_and_offsets,
)
from research.peak_rank_detection.train_synthetic_real_detector import (
    SELECTION_INDICES,
    TrainingExample,
    load_synthetic_examples,
    normalize_frames,
    sequence_paths,
)


RUN_ID = "synthetic-complete-global-peak-threshold-v1"
THRESHOLD_POLICY = "synthetic_selection_micro_detection_jaccard"
DEFAULT_TTA_MODES = ("none", "zflip2", "rot4", "d4")
MATCH_RADIUS = 2.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def scored_matches(
    predicted: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    *,
    radius: float = MATCH_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return descending scores and one-to-one TP flags for one labeled crop."""

    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 3)
    if len(predicted) != len(scores) or not len(truth):
        raise ValueError("calibration predictions must align and truth must be nonempty")
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    true_positive = np.zeros(len(order), dtype=bool)
    matched: set[int] = set()
    for rank, prediction_index in enumerate(order):
        distances = np.linalg.norm(truth - predicted[int(prediction_index)], axis=1)
        for truth_index in np.argsort(distances, kind="stable"):
            if float(distances[int(truth_index)]) > radius:
                break
            if int(truth_index) not in matched:
                matched.add(int(truth_index))
                true_positive[rank] = True
                break
    return ordered_scores, true_positive


def select_global_threshold(
    rows: list[tuple[np.ndarray, np.ndarray]], *, total_truth: int
) -> dict[str, Any]:
    """Maximize micro TP/(TP+FP+FN), with conservative deterministic ties."""

    if total_truth <= 0 or not rows:
        raise ValueError("threshold calibration requires labeled predictions")
    scores = np.concatenate([row[0] for row in rows])
    true_positive = np.concatenate([row[1] for row in rows])
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("threshold calibration produced no finite peak scores")
    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    true_positive = true_positive[order]
    cumulative_tp = np.cumsum(true_positive, dtype=np.int64)
    selected = np.arange(1, len(scores) + 1, dtype=np.int64)
    cumulative_fp = selected - cumulative_tp
    jaccard = cumulative_tp / (float(total_truth) + cumulative_fp)
    # Only score after the last occurrence of a tied probability.  np.argmax
    # then provides the fewest-predictions tie break at equal Jaccard.
    group_end = np.r_[scores[1:] != scores[:-1], True]
    eligible = np.flatnonzero(group_end)
    best = int(eligible[int(np.argmax(jaccard[eligible]))])
    boundary = float(scores[best])
    if best + 1 < len(scores):
        threshold = (boundary + float(scores[best + 1])) / 2.0
    else:
        # The calibration set does not identify the behavior below its lowest
        # observed selected peak.  Stay immediately below that boundary rather
        # than lowering all the way to the extraction floor.
        threshold = float(np.nextafter(boundary, 0.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    tp = int(cumulative_tp[best])
    predictions = best + 1
    fp = predictions - tp
    fn = int(total_truth) - tp
    return {
        "threshold": threshold,
        "selected_predictions": predictions,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / predictions,
        "recall": tp / int(total_truth),
        "detection_jaccard": tp / (tp + fp + fn),
        "candidate_peaks": int(len(scores)),
        "minimum_candidate_probability": LOW_PROBABILITY_THRESHOLD,
        "match_radius_voxels": MATCH_RADIUS,
    }


@torch.inference_mode()
def predict_example(
    model: torch.nn.Module,
    example: TrainingExample,
    *,
    device: torch.device,
    tta_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    frames = torch.from_numpy(normalize_frames(example.frames))[None].to(device)
    probability, offsets = predict_probability_and_offsets(
        model, frames, tta_mode=tta_mode
    )
    return peaks_from_prediction(probability[0, 0], offsets[0])


def calibrate_mode(
    model: torch.nn.Module,
    examples: list[TrainingExample],
    *,
    device: torch.device,
    tta_mode: str,
    started: float,
    max_wall_seconds: float,
) -> dict[str, Any]:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    example_rows = []
    total_truth = 0
    for example in examples:
        predicted, scores = predict_example(
            model, example, device=device, tta_mode=tta_mode
        )
        ordered_scores, true_positive = scored_matches(
            predicted, scores, example.points
        )
        rows.append((ordered_scores, true_positive))
        total_truth += len(example.points)
        example_rows.append(
            {
                "identity": example.identity,
                "truth_nodes": int(len(example.points)),
                "candidate_peaks": int(len(scores)),
            }
        )
        if time.monotonic() - started > max_wall_seconds:
            raise TimeoutError("clean threshold calibration exceeded its wall guard")
    selected = select_global_threshold(rows, total_truth=total_truth)
    return {
        **selected,
        "tta_mode": tta_mode,
        "tta_views": len(TTA_TRANSFORMS[tta_mode]),
        "examples": example_rows,
        "complete_synthetic_examples": len(examples),
        "total_truth_nodes": int(total_truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tta-modes", default=",".join(DEFAULT_TTA_MODES))
    parser.add_argument("--max-wall-seconds", type=float, default=3_600.0)
    args = parser.parse_args()
    modes = tuple(mode.strip() for mode in args.tta_modes.split(",") if mode.strip())
    if (
        modes != DEFAULT_TTA_MODES
        or args.max_wall_seconds <= 0
        or args.output.exists()
    ):
        raise ValueError("calibration modes, wall guard, or output contract changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("threshold calibration requires exactly one visible CUDA GPU")

    started = time.monotonic()
    terminal = validate_training(args.checkpoint, args.training_terminal)
    device = torch.device("cuda:0")
    model = load_model(args.checkpoint, terminal, device)
    paths = sequence_paths(args.synthetic_root)
    examples = load_synthetic_examples(paths, SELECTION_INDICES)
    if len(examples) != 24:
        raise RuntimeError("synthetic selection inventory changed")
    source_rows = [
        {
            "index": int(index),
            "name": paths[int(index)].name,
            "sha256": sha256_file(paths[int(index)]),
        }
        for index in SELECTION_INDICES
    ]
    calibrations = {
        mode: calibrate_mode(
            model,
            examples,
            device=device,
            tta_mode=mode,
            started=started,
            max_wall_seconds=args.max_wall_seconds,
        )
        for mode in modes
    }
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "calibrated",
        "threshold_policy": THRESHOLD_POLICY,
        "thresholds": calibrations,
        "selection_indices": list(SELECTION_INDICES),
        "selection_sources": source_rows,
        "complete_synthetic_labels_read": True,
        "competition_train_data_read": False,
        "competition_test_data_read": False,
        "organizer_estimated_node_count_read": False,
        "organizer_estimated_node_count_used_for_threshold": False,
        "public_predictions_read": False,
        "public_notebook_weights_read": False,
        "public_leaderboard_used_for_selection": False,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "training_terminal_sha256": sha256_file(args.training_terminal),
        "parameter_count": terminal["parameter_count"],
        "elapsed_seconds": time.monotonic() - started,
    }
    if any(
        not 0.0 < float(row["threshold"]) < 1.0
        or not math.isfinite(float(row["detection_jaccard"]))
        for row in calibrations.values()
    ):
        raise RuntimeError("calibrated threshold contract is invalid")
    atomic_json(args.output, payload)
    print("PEAK THRESHOLD CALIBRATION", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
