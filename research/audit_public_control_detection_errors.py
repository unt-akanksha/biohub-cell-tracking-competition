#!/usr/bin/env python
"""Classify residual node errors in the frozen public-control predictions.

This is a diagnostic over the already-opened four acceptance movies.  It does
not select a model, threshold, or submission policy.  Misses are separated into
one-to-one crowding conflicts, local localization errors, and absent detections
so the next detector experiment addresses the observed failure mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.spotiflow_biohub.evaluate_pretrained_detector import (
    ACCEPTANCE_STEMS,
    VOXEL_SCALE_UM,
    graph_points_by_frame,
)


RUN_ID = "public-control-detection-error-audit-v1"
MATCH_RADIUS_UM = 5.0
LOCALIZATION_RADIUS_UM = 12.0
DENSITY_RADIUS_UM = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], value: float) -> float | None:
    return float(np.percentile(values, value)) if values else None


def classify_frame(
    predicted_voxel: np.ndarray,
    truth_voxel: np.ndarray,
    *,
    match_radius_um: float = MATCH_RADIUS_UM,
    localization_radius_um: float = LOCALIZATION_RADIUS_UM,
    density_radius_um: float = DENSITY_RADIUS_UM,
) -> list[dict[str, Any]]:
    predicted = np.asarray(predicted_voxel, dtype=np.float64).reshape(-1, 3)
    truth = np.asarray(truth_voxel, dtype=np.float64).reshape(-1, 3)
    if min(match_radius_um, localization_radius_um, density_radius_um) <= 0:
        raise ValueError("detection audit radii must be positive")
    if localization_radius_um <= match_radius_um:
        raise ValueError("localization radius must exceed the match radius")
    predicted_um = predicted * VOXEL_SCALE_UM
    truth_um = truth * VOXEL_SCALE_UM
    distances = np.linalg.norm(
        predicted_um[:, None, :] - truth_um[None, :, :], axis=-1
    ) if len(predicted) and len(truth) else np.empty((len(predicted), len(truth)))
    matched_truth: set[int] = set()
    if distances.size:
        rows, columns = linear_sum_assignment(distances)
        matched_truth = {
            int(column)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
            if distances[row, column] <= match_radius_um
        }
    truth_pairwise = (
        np.linalg.norm(truth_um[:, None, :] - truth_um[None, :, :], axis=-1)
        if len(truth)
        else np.empty((0, 0))
    )
    if len(truth_pairwise):
        np.fill_diagonal(truth_pairwise, np.inf)
    rows = []
    for index in range(len(truth)):
        nearest = float(distances[:, index].min()) if len(predicted) else float("inf")
        matched = index in matched_truth
        if matched:
            failure = "matched"
        elif nearest <= match_radius_um:
            failure = "crowding_conflict"
        elif nearest <= localization_radius_um:
            failure = "localization_miss"
        else:
            failure = "absent_detection"
        rows.append(
            {
                "truth_index": index,
                "matched": matched,
                "failure_mode": failure,
                "nearest_prediction_um": nearest,
                "predictions_within_density_radius": int(
                    np.count_nonzero(distances[:, index] <= density_radius_um)
                ) if len(predicted) else 0,
                "nearest_annotated_neighbor_um": (
                    float(truth_pairwise[index].min()) if len(truth) > 1 else None
                ),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        name: sum(row["failure_mode"] == name for row in rows)
        for name in (
            "matched",
            "crowding_conflict",
            "localization_miss",
            "absent_detection",
        )
    }
    misses = [row for row in rows if not row["matched"]]
    finite_miss_distances = [
        float(row["nearest_prediction_um"])
        for row in misses
        if np.isfinite(float(row["nearest_prediction_um"]))
    ]
    matched = counts["matched"]
    total = len(rows)
    return {
        "annotated_nodes": total,
        "matched_nodes": matched,
        "annotated_node_recall": matched / total if total else 0.0,
        "failure_counts": counts,
        "miss_nearest_prediction_um_p50": percentile(finite_miss_distances, 50),
        "miss_nearest_prediction_um_p90": percentile(finite_miss_distances, 90),
        "miss_mean_predictions_within_10um": (
            float(np.mean([row["predictions_within_density_radius"] for row in misses]))
            if misses
            else 0.0
        ),
        "matched_mean_predictions_within_10um": (
            float(
                np.mean(
                    [
                        row["predictions_within_density_radius"]
                        for row in rows
                        if row["matched"]
                    ]
                )
            )
            if matched
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    movie_rows = []
    all_rows = []
    source_hashes = {}
    for stem in ACCEPTANCE_STEMS:
        prediction = args.prediction_root / f"{stem}.geff"
        truth = args.truth_root / f"{stem}.geff"
        if not prediction.is_dir() or not truth.is_dir():
            raise FileNotFoundError(f"missing detection-audit graph for {stem}")
        predicted_by_frame = graph_points_by_frame(prediction)
        truth_by_frame = graph_points_by_frame(truth)
        rows = []
        for frame, truth_points in sorted(truth_by_frame.items()):
            classified = classify_frame(
                predicted_by_frame.get(frame, np.empty((0, 3))), truth_points
            )
            for row in classified:
                row.update({"stem": stem, "frame": int(frame)})
            rows.extend(classified)
        movie_rows.append({"stem": stem, **summarize_rows(rows)})
        all_rows.extend(rows)
        source_hashes[stem] = {
            "prediction_geff_sha256": hashlib.sha256(
                "\n".join(
                    f"{path.relative_to(prediction).as_posix()}:{sha256_file(path)}"
                    for path in sorted(prediction.rglob("*"))
                    if path.is_file()
                ).encode("utf-8")
            ).hexdigest(),
            "truth_geff_sha256": hashlib.sha256(
                "\n".join(
                    f"{path.relative_to(truth).as_posix()}:{sha256_file(path)}"
                    for path in sorted(truth.rglob("*"))
                    if path.is_file()
                ).encode("utf-8")
            ).hexdigest(),
        }
    result = {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "run_id": RUN_ID,
        "radii_um": {
            "match": MATCH_RADIUS_UM,
            "localization": LOCALIZATION_RADIUS_UM,
            "density": DENSITY_RADIUS_UM,
        },
        "movies": movie_rows,
        "pooled": summarize_rows(all_rows),
        "source_hashes": source_hashes,
        "acceptance_labels_already_opened": True,
        "model_or_threshold_selected": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "authorized_for_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
