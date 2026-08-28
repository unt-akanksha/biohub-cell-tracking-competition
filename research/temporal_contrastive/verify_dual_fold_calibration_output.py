#!/usr/bin/env python
"""Fail-closed audit of downloaded two-GPU contextual calibration output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

try:
    from research.trackastra_graph.verify_dual_fold_training_output import (
        terminal_source_policy,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research.trackastra_graph.verify_dual_fold_training_output import (
        terminal_source_policy,
    )


RUN_ID = "temporal-contextual-pair-fusion-blend-v3"
APPEARANCE_RUN_ID = "temporal-contextual-pair-fusion-v3"
APPEARANCE_FAMILY = "temporal_contextual_pair_fusion_v3"
TRACKASTRA_RUN_ID = "trackastra-dual-fold-synthetic-v1"
FOLDS = ("target_44b6", "target_6bba")
PREFIX_BY_FOLD = {"target_44b6": "44b6", "target_6bba": "6bba"}
OPENED_ACCEPTANCE_STEMS = frozenset(
    {
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)
ENSEMBLE_MODES = ("target_only", "reciprocal_mean")
APPEARANCE_WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.35)
DIVISION_WEIGHTS = (0.0, 0.05, 0.10, 0.20)
APPEARANCE_TEMPERATURE = 0.10
MINIMUM_POOLED_GAIN = 0.001
MAXIMUM_MOVIE_REGRESSION = 0.002
FROZEN_LINK_CONFIGURATION = {
    "edge_threshold": 0.08,
    "division_threshold": 0.18,
    "division_ratio": 0.50,
}
COUNT_KEYS = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "predicted_edges",
    "true_edges",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def same_number(left: Any, right: Any) -> bool:
    return math.isclose(
        finite_number(left, "recorded metric"),
        finite_number(right, "recomputed metric"),
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def _checked_count(row: Mapping[str, Any], key: str, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} has an invalid {key}")
    return value


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return float(tp / denominator) if denominator else 1.0


def _verify_metric_row(row: Mapping[str, Any], *, label: str) -> dict[str, int]:
    counts = {key: _checked_count(row, key, label) for key in COUNT_KEYS}
    edge = _jaccard(counts["edge_tp"], counts["edge_fp"], counts["edge_fn"])
    division = _jaccard(
        counts["division_tp"], counts["division_fp"], counts["division_fn"]
    )
    expected = {
        "edge_jaccard": edge,
        "adjusted_edge_jaccard": edge,
        "division_jaccard": division,
        "composite": edge + 0.10 * division,
    }
    if any(not same_number(row.get(key), value) for key, value in expected.items()):
        raise ValueError(f"{label} metrics do not match confusion counts")
    return counts


def _verify_grid_row(
    row: Mapping[str, Any], *, stems: set[str], fold: str
) -> tuple[tuple[str, float, float], dict[str, float]]:
    mode = str(row.get("ensemble_mode", ""))
    appearance_weight = finite_number(row.get("appearance_weight"), "appearance weight")
    division_weight = finite_number(row.get("division_weight"), "division weight")
    key = (mode, appearance_weight, division_weight)
    by_movie = row.get("by_movie")
    if not isinstance(by_movie, list) or len(by_movie) != len(stems):
        raise ValueError(f"calibration movie inventory changed: {fold}")
    movie_scores: dict[str, float] = {}
    totals = {name: 0 for name in COUNT_KEYS}
    for movie in by_movie:
        if not isinstance(movie, dict):
            raise ValueError(f"calibration movie row is malformed: {fold}")
        stem = str(movie.get("stem", ""))
        if not stem or stem in movie_scores:
            raise ValueError(f"calibration movie row is duplicated: {fold}")
        counts = _verify_metric_row(movie, label=f"{fold}/{stem}")
        movie_scores[stem] = finite_number(movie.get("composite"), "movie composite")
        for name, value in counts.items():
            totals[name] += value
    if set(movie_scores) != stems:
        raise ValueError(f"calibration movie inventory diverges: {fold}")

    pooled = row.get("pooled")
    if not isinstance(pooled, dict):
        raise ValueError(f"pooled calibration metrics are missing: {fold}")
    pooled_counts = _verify_metric_row(pooled, label=f"{fold}/pooled")
    if pooled_counts != totals or pooled.get("movies") != len(stems):
        raise ValueError(f"pooled calibration metrics do not aggregate movies: {fold}")
    return key, movie_scores


def _recompute_selection(selection: Mapping[str, Any], *, stems: set[str], fold: str) -> dict[str, Any]:
    grid = selection.get("grid")
    if not isinstance(grid, list) or len(grid) != 40:
        raise ValueError(f"calibration grid does not contain 40 rows: {fold}")
    expected_keys = {
        (mode, appearance_weight, division_weight)
        for mode in ENSEMBLE_MODES
        for appearance_weight in APPEARANCE_WEIGHTS
        for division_weight in DIVISION_WEIGHTS
    }
    rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    movie_scores: dict[tuple[str, float, float], dict[str, float]] = {}
    for raw in grid:
        if not isinstance(raw, dict):
            raise ValueError(f"calibration grid row is malformed: {fold}")
        key, scores = _verify_grid_row(raw, stems=stems, fold=fold)
        if key in rows:
            raise ValueError(f"calibration grid contains duplicate weights: {fold}")
        rows[key] = raw
        movie_scores[key] = scores
    if set(rows) != expected_keys:
        raise ValueError(f"calibration grid is incomplete: {fold}")

    control_key = ("target_only", 0.0, 0.0)
    control = rows[control_key]
    control_scores = movie_scores[control_key]
    control_composite = finite_number(
        control.get("pooled", {}).get("composite"), "control pooled composite"
    )
    eligible_rows: list[dict[str, Any]] = []
    for key, row in rows.items():
        deltas = {
            stem: movie_scores[key][stem] - control_scores[stem] for stem in stems
        }
        pooled_gain = (
            finite_number(row.get("pooled", {}).get("composite"), "pooled composite")
            - control_composite
        )
        worst_delta = min(deltas.values())
        eligible = bool(
            key[1] > 0.0
            and pooled_gain >= MINIMUM_POOLED_GAIN
            and worst_delta >= -MAXIMUM_MOVIE_REGRESSION
        )
        recorded_deltas = row.get("movie_deltas_vs_zero")
        if not isinstance(recorded_deltas, dict) or set(recorded_deltas) != stems:
            raise ValueError(f"movie delta inventory changed: {fold}")
        if any(not same_number(recorded_deltas[stem], value) for stem, value in deltas.items()):
            raise ValueError(f"movie deltas were not recomputed faithfully: {fold}")
        if not (
            same_number(row.get("pooled_gain_vs_zero"), pooled_gain)
            and same_number(row.get("worst_movie_delta_vs_zero"), worst_delta)
            and row.get("eligible") is eligible
        ):
            raise ValueError(f"calibration eligibility evidence diverges: {fold}")
        if eligible:
            eligible_rows.append(row)

    if not eligible_rows:
        expected_selected = control
    else:
        expected_selected = max(
            eligible_rows,
            key=lambda row: (
                finite_number(row["worst_movie_delta_vs_zero"], "worst delta"),
                finite_number(row["pooled_gain_vs_zero"], "pooled gain"),
                1 if row["ensemble_mode"] == "target_only" else 0,
                -finite_number(row["division_weight"], "division weight"),
                -finite_number(row["appearance_weight"], "appearance weight"),
            ),
        )
    if selection.get("control") != control or selection.get("selected") != expected_selected:
        raise ValueError(f"recorded calibration selection is not the grid winner: {fold}")
    if not (
        same_number(selection.get("selected_weight"), expected_selected["appearance_weight"])
        and same_number(
            selection.get("selected_division_weight"),
            expected_selected["division_weight"],
        )
        and selection.get("selected_ensemble_mode") == expected_selected["ensemble_mode"]
        and selection.get("improved") is bool(expected_selected["eligible"])
    ):
        raise ValueError(f"recorded calibration selection metadata diverges: {fold}")
    return expected_selected


def _verify_sources(
    appearance_root: Path, trackastra_root: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    appearance = read_json(appearance_root / "training_terminal.json")
    trackastra = read_json(trackastra_root / "training_terminal.json")
    expected_folds = set(FOLDS)
    if not (
        appearance.get("schema_version") == 1
        and appearance.get("status") == "completed"
        and appearance.get("run_id") == APPEARANCE_RUN_ID
        and appearance.get("appearance_family") == APPEARANCE_FAMILY
        and appearance.get("gpu_count") == 2
        and appearance.get("both_folds_trained") is True
        and appearance.get("both_folds_improved") is True
        and appearance.get("public_predictions_copied") is False
        and appearance.get("public_leaderboard_used_for_selection") is False
        and appearance.get("submission_created") is False
        and set(appearance.get("folds", {})) == expected_folds
    ):
        raise ValueError("appearance calibration source is not eligible")
    if not (
        trackastra.get("schema_version") == 1
        and trackastra.get("status") == "completed"
        and trackastra.get("run_id") == TRACKASTRA_RUN_ID
        and trackastra.get("gpu_count") == 2
        and trackastra.get("submission_created") is False
        and set(trackastra.get("folds", {})) == expected_folds
    ):
        raise ValueError("Trackastra calibration source is not eligible")
    source_policy = terminal_source_policy(trackastra)
    if source_policy not in {"adapted_dual_fold", "predeclared_pretrained_control"}:
        raise ValueError("Trackastra source policy is not eligible")

    for fold in FOLDS:
        appearance_worker = appearance["folds"][fold]
        trackastra_worker = trackastra["folds"][fold]
        if read_json(appearance_root / fold / "worker_terminal.json") != appearance_worker:
            raise ValueError(f"appearance aggregate and worker diverge: {fold}")
        if read_json(trackastra_root / fold / "worker_terminal.json") != trackastra_worker:
            raise ValueError(f"Trackastra aggregate and worker diverge: {fold}")
        appearance_model = appearance_root / fold / "appearance_model.pt"
        trackastra_model = trackastra_root / fold / "model.pt"
        if sha256_file(appearance_model) != appearance_worker.get("model_sha256"):
            raise ValueError(f"appearance checkpoint hash mismatch: {fold}")
        if sha256_file(trackastra_model) != trackastra_worker.get("model_sha256"):
            raise ValueError(f"Trackastra checkpoint hash mismatch: {fold}")
    return appearance, trackastra, source_policy


def verify_output(
    root: Path, *, appearance_root: Path, trackastra_root: Path
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    appearance_root = appearance_root.expanduser().resolve()
    trackastra_root = trackastra_root.expanduser().resolve()
    launcher_path = root / "calibration_launcher_terminal.json"
    calibration_root = root / "temporal_contextual_calibration_v3"
    aggregate_path = calibration_root / "calibration_terminal.json"
    launcher = read_json(launcher_path)
    aggregate = read_json(aggregate_path)
    appearance, trackastra, source_policy = _verify_sources(
        appearance_root, trackastra_root
    )
    aggregate_sha256 = sha256_file(aggregate_path)
    if not (
        launcher.get("schema_version") == 1
        and launcher.get("status") == "completed"
        and launcher.get("run_id") == RUN_ID
        and launcher.get("declared_budget_seconds") == 21_600
        and launcher.get("calibrator_hard_stop_seconds") == 19_800
        and launcher.get("calibration_terminal_exists") is True
        and launcher.get("calibration_terminal_sha256") == aggregate_sha256
        and launcher.get("gpu_count_required") == 2
        and launcher.get("processed_acceptance_ground_truth_read") is False
        and launcher.get("public_leaderboard_used_for_selection") is False
        and launcher.get("submission_created") is False
    ):
        raise ValueError("calibration launcher terminal is not eligible")
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("status") == "completed"
        and aggregate.get("run_id") == RUN_ID
        and aggregate.get("appearance_family") == APPEARANCE_FAMILY
        and aggregate.get("gpu_count") == 2
        and aggregate.get("both_folds_improved") is True
        and aggregate.get("processed_acceptance_ground_truth_read") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and aggregate.get("submission_created") is False
        and set(aggregate.get("folds", {})) == set(FOLDS)
    ):
        raise ValueError("aggregate calibration terminal is not eligible")

    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ("submission" in path.name.lower() or path.suffix.lower() in {".csv", ".zip"})
    ]
    if forbidden:
        raise ValueError(f"calibration output contains competition artifacts: {forbidden}")

    selected: dict[str, dict[str, Any]] = {}
    calibration_stems: dict[str, set[str]] = {}
    for fold in FOLDS:
        fold_path = calibration_root / fold / "calibration_result.json"
        row = read_json(fold_path)
        if row != aggregate["folds"].get(fold):
            raise ValueError(f"aggregate and calibration worker diverge: {fold}")
        stems_raw = row.get("calibration_stems")
        if not isinstance(stems_raw, list) or len(stems_raw) != 12:
            raise ValueError(f"calibration stem inventory changed: {fold}")
        stems = set(map(str, stems_raw))
        peer_fold = next(candidate for candidate in FOLDS if candidate != fold)
        appearance_worker = appearance["folds"][fold]
        peer_worker = appearance["folds"][peer_fold]
        trackastra_worker = trackastra["folds"][fold]
        appearance_config = read_json(appearance_root / fold / "training_config.json")
        if not (
            len(stems) == 12
            and stems == set(map(str, appearance_config.get("real_calibration_stems_reserved", [])))
            and not stems & OPENED_ACCEPTANCE_STEMS
            and all(stem.startswith(f"{PREFIX_BY_FOLD[fold]}_") for stem in stems)
            and row.get("schema_version") == 1
            and row.get("status") == "completed"
            and row.get("run_id") == RUN_ID
            and row.get("appearance_family") == APPEARANCE_FAMILY
            and row.get("fold") == fold
            and row.get("peer_fold") == peer_fold
            and row.get("appearance_model_sha256") == appearance_worker.get("model_sha256")
            and row.get("peer_appearance_model_sha256") == peer_worker.get("model_sha256")
            and row.get("trackastra_model_sha256") == trackastra_worker.get("model_sha256")
            and row.get("trackastra_source_policy") == source_policy
            and row.get("ensemble_mode_grid") == list(ENSEMBLE_MODES)
            and row.get("appearance_weight_grid") == list(APPEARANCE_WEIGHTS)
            and row.get("division_weight_grid") == list(DIVISION_WEIGHTS)
            and same_number(row.get("appearance_temperature"), APPEARANCE_TEMPERATURE)
            and row.get("frozen_clean_link_configuration") == FROZEN_LINK_CONFIGURATION
            and same_number(row.get("minimum_pooled_gain"), MINIMUM_POOLED_GAIN)
            and same_number(row.get("maximum_movie_regression"), MAXIMUM_MOVIE_REGRESSION)
            and row.get("processed_acceptance_ground_truth_read") is False
            and row.get("public_leaderboard_used_for_selection") is False
            and row.get("submission_created") is False
        ):
            raise ValueError(f"calibration fold contract changed: {fold}")
        selection = row.get("selection")
        if not isinstance(selection, dict):
            raise ValueError(f"calibration selection is missing: {fold}")
        winner = _recompute_selection(selection, stems=stems, fold=fold)
        if selection.get("improved") is not True or winner.get("eligible") is not True:
            raise ValueError(f"calibration fold did not improve clean controls: {fold}")
        selected[fold] = {
            "ensemble_mode": winner["ensemble_mode"],
            "appearance_weight": winner["appearance_weight"],
            "division_weight": winner["division_weight"],
            "pooled_gain_vs_zero": winner["pooled_gain_vs_zero"],
            "worst_movie_delta_vs_zero": winner["worst_movie_delta_vs_zero"],
            "calibration_result_sha256": sha256_file(fold_path),
        }
        calibration_stems[fold] = stems
    if calibration_stems[FOLDS[0]] & calibration_stems[FOLDS[1]]:
        raise ValueError("reciprocal calibration stems overlap")

    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "appearance_family": APPEARANCE_FAMILY,
        "gpu_count": 2,
        "root": str(root),
        "selected": selected,
        "source_policy": source_policy,
        "calibration_terminal_sha256": aggregate_sha256,
        "calibration_launcher_terminal_sha256": sha256_file(launcher_path),
        "appearance_training_terminal_sha256": sha256_file(
            appearance_root / "training_terminal.json"
        ),
        "trackastra_training_terminal_sha256": sha256_file(
            trackastra_root / "training_terminal.json"
        ),
        "competition_artifacts_found": False,
        "authorized_for_processed_materialization": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--appearance-root", type=Path, required=True)
    parser.add_argument("--trackastra-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_output(
                args.root,
                appearance_root=args.appearance_root,
                trackastra_root=args.trackastra_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
