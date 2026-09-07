#!/usr/bin/env python
"""Score a materialized peak-rank candidate with the pinned official metric.

The Kaggle candidate notebook writes the exact integer-coordinate graphs used
for its held-out validation.  This CPU-only step runs after those outputs are
downloaded, verifies the scorer and frozen control bytes, and compares all four
complete movies.  It never reads competition test labels or submits anything.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import polars as pl

from biohub_tracker.graphs import artifact_tree_sha256, load_geff_graph
from biohub_tracker.io import atomic_write_json, sha256_file
from biohub_tracker.scorer_lock import verify_scorer_lock
from biohub_tracker.submission_io import _load_pinned_script


RUN_ID = "peak-rank-patched-official-complete-movie-v1"
EVALUATION_KIND = "patched_official_complete_movie_candidate_vs_frozen_control"
EXPECTED_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
EXPECTED_SCORER_LOCK_SHA256 = (
    "1db65dee620059f19bf16633aa54a9f3379eb5d5bdff148a4037b949393b7a9c"
)
EXPECTED_CONTROL_SCORE = 0.9343483108193262
MINIMUM_SCORE_GAIN = 0.003
MAXIMUM_ADJUSTED_EDGE_REGRESSION = 0.001
MAXIMUM_MOVIE_SCORE_REGRESSION = 0.005
FLOAT_TOLERANCE = 1e-12
CSV_COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
CONTROL_TREE_SHA256 = {
    "44b6_12dfb391": "9f127687bf19a30a89caafadfb1130b7c94ef1a8b4c98d24371d4d9c095378fa",
    "44b6_267148e4": "7f34798fe32ebe47b892b70b9b3769a82cf38806f40d106574c2cc7f47dccb78",
    "6bba_062c8d37": "242c98b4499fdcdde9d124895d08d2784f9a1488c90c6d08427a37f1b9afa1c1",
    "6bba_07e24132": "c5d87fa70bc2520e88581c1dbd4201f6920eb2a7f26a2643630e289da6993914",
}
TRUTH_TREE_SHA256 = {
    "44b6_12dfb391": "3ab99b7745872f50d0f55532c7284a7e648c617bfa9cf4bf203af6387b2c41ca",
    "44b6_267148e4": "983acc7565e54cb51693e8bd855a23836168bbbe241f49828d838cf3d0712eab",
    "6bba_062c8d37": "c7dc6c4178cd63ef5abb77e852bd048ce515b0313434bfb07e6188094fc66048",
    "6bba_07e24132": "b0bf2e5c4380c972f062ff8361ce8ec2523730d6d2257b52df66793bd435c456",
}
EXPECTED_CONTROL_COUNTS = {
    "44b6_12dfb391": (744, 36, 29, 0, 0, 1, 44_139),
    "44b6_267148e4": (254, 34, 23, 0, 2, 1, 21_843),
    "6bba_062c8d37": (883, 8, 15, 1, 0, 0, 5_817),
    "6bba_07e24132": (312, 20, 33, 0, 1, 2, 26_172),
}
COUNT_NAMES = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "num_pred_nodes",
)


def finite_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def read_truth_metadata(path: Path) -> tuple[tuple[float, float, float], int]:
    payload = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    geff = payload["attributes"]["geff"]
    axes = {axis["name"]: axis for axis in geff["axes"]}
    scale = tuple(float(axes[name]["scale"]) for name in ("z", "y", "x"))
    estimated = int(geff["extra"]["estimated_number_of_nodes"])
    if any(not math.isfinite(value) or value <= 0 for value in scale) or estimated <= 0:
        raise ValueError(f"invalid truth scoring metadata: {path}")
    return scale, estimated


def load_candidate_table(path: Path) -> pl.DataFrame:
    table = pl.read_csv(path)
    if tuple(table.columns) != CSV_COLUMNS or table.height == 0:
        raise ValueError("official candidate validator CSV schema is invalid")
    if table.get_column("id").to_list() != list(range(table.height)):
        raise ValueError("official candidate validator row ids are not contiguous")
    stems = set(table.get_column("dataset").unique().to_list())
    if stems != set(EXPECTED_STEMS):
        raise ValueError(
            f"candidate validator coverage mismatch: expected {EXPECTED_STEMS}, saw {sorted(stems)}"
        )
    invalid_types = set(table.get_column("row_type").unique().to_list()) - {
        "node",
        "edge",
    }
    if invalid_types:
        raise ValueError(f"invalid candidate validator row types: {sorted(invalid_types)}")
    node_rows = table.filter(pl.col("row_type") == "node")
    for name in ("node_id", "t", "z", "y", "x"):
        values = node_rows.get_column(name).cast(pl.Float64)
        if not values.is_finite().all() or not (values == values.round(0)).all():
            raise ValueError(f"candidate validator has nonintegral {name}")
    return table


def score_graph(
    verified: Any,
    prediction: Any,
    truth: Any,
    *,
    stem: str,
    scale: tuple[float, float, float],
    estimated_nodes: int,
) -> dict[str, Any]:
    scored_prediction = prediction.copy()
    scored_truth = truth.copy()
    result = verified.evaluate(
        scored_prediction,
        scored_truth,
        scale=scale,
        max_distance=float(verified.lock.raw["constants"]["max_distance_um"]),
    )
    recall = float(verified.node_recall(scored_prediction, scored_truth))
    organizer_row = dict(
        verified.per_sample_metrics(result, float(estimated_nodes), recall)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No divisions present across any sample in this split*"
        )
        summary = dict(verified.summarise([organizer_row]))
    counts = {name: int(getattr(result, name)) for name in COUNT_NAMES}
    return {
        "sample_id": stem,
        "organizer_row": organizer_row,
        "official_counts": counts,
        "gt_node_count": int(scored_truth.num_nodes()),
        "matched_gt_node_count": round(recall * scored_truth.num_nodes()),
        "node_recall": recall,
        "edge_jaccard": float(organizer_row["edge_jaccard"]),
        "adjusted_edge_jaccard": float(organizer_row["adj_edge_jaccard"]),
        "division_jaccard": finite_float(summary["division_jaccard"]),
        "score": float(summary["score"]),
    }


def aggregate(verified: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No divisions present across any sample in this split*"
        )
        summary = dict(
            verified.summarise([dict(row["organizer_row"]) for row in rows])
        )
    counts = {
        name: sum(int(row["official_counts"][name]) for row in rows)
        for name in COUNT_NAMES
    }
    matched = sum(int(row["matched_gt_node_count"]) for row in rows)
    gt_nodes = sum(int(row["gt_node_count"]) for row in rows)
    return {
        "movie_count": len(rows),
        "official_counts": counts,
        "matched_gt_node_count": matched,
        "gt_node_count": gt_nodes,
        "node_recall_micro": matched / gt_nodes,
        "organizer_macro_node_recall": float(summary["node_recall"]),
        "edge_jaccard": float(summary["edge_jaccard"]),
        "adjusted_edge_jaccard": float(summary["adj_edge_jaccard"]),
        "division_jaccard": finite_float(summary["division_jaccard"]),
        "score": float(summary["score"]),
    }


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "organizer_row"}


def metric_delta(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in (
        "score",
        "adjusted_edge_jaccard",
        "edge_jaccard",
        "division_jaccard",
        "node_recall_micro",
    ):
        left, right = candidate.get(name), control.get(name)
        result[name] = None if left is None or right is None else float(left) - float(right)
    return result


def exact_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    by_movie: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    movie_deltas = {
        str(row["sample_id"]): float(row["score_delta"]) for row in by_movie
    }
    embryo_deltas = {
        prefix: sum(
            delta for stem, delta in movie_deltas.items() if stem.startswith(prefix + "_")
        )
        / sum(stem.startswith(prefix + "_") for stem in movie_deltas)
        for prefix in ("44b6", "6bba")
    }
    score_gain = float(candidate["score"]) - float(control["score"])
    adjusted_edge_delta = float(candidate["adjusted_edge_jaccard"]) - float(
        control["adjusted_edge_jaccard"]
    )
    worst_movie_delta = min(movie_deltas.values())
    checks = {
        "all_complete_movies_scored": set(movie_deltas) == set(EXPECTED_STEMS),
        "control_score_reproduced": math.isclose(
            float(control["score"]),
            EXPECTED_CONTROL_SCORE,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ),
        "minimum_pooled_score_gain_passed": score_gain >= MINIMUM_SCORE_GAIN,
        "adjusted_edge_regression_floor_passed": (
            adjusted_edge_delta >= -MAXIMUM_ADJUSTED_EDGE_REGRESSION
        ),
        "worst_movie_regression_floor_passed": (
            worst_movie_delta >= -MAXIMUM_MOVIE_SCORE_REGRESSION
        ),
        "candidate_not_exact_control_replica": any(
            abs(delta) > FLOAT_TOLERANCE for delta in movie_deltas.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_score_gain": MINIMUM_SCORE_GAIN,
        "maximum_adjusted_edge_regression": MAXIMUM_ADJUSTED_EDGE_REGRESSION,
        "maximum_movie_score_regression": MAXIMUM_MOVIE_SCORE_REGRESSION,
        "score_gain": score_gain,
        "adjusted_edge_delta": adjusted_edge_delta,
        "worst_movie_score_delta": worst_movie_delta,
        "score_delta_by_movie": movie_deltas,
        "mean_score_delta_by_embryo": embryo_deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-validator", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--scorer-lock", type=Path, required=True)
    parser.add_argument("--organizer-checkout", type=Path, required=True)
    parser.add_argument("--tracksdata-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite exact evidence: {args.output}")
    candidate_sha256 = sha256_file(args.candidate_validator)
    candidate_table = load_candidate_table(args.candidate_validator)
    verified = verify_scorer_lock(
        args.scorer_lock,
        args.organizer_checkout,
        tracksdata_checkout=args.tracksdata_checkout,
    )
    if verified.lock_sha256 != EXPECTED_SCORER_LOCK_SHA256:
        raise RuntimeError("official scorer lock identity changed")
    rebuilder = _load_pinned_script(verified, "csv_to_geffs.py")
    rows: dict[str, list[dict[str, Any]]] = {"control": [], "candidate": []}
    control_hashes: dict[str, str] = {}
    truth_hashes: dict[str, str] = {}
    for stem in EXPECTED_STEMS:
        control_path = args.control_dir / f"{stem}.geff"
        truth_path = args.truth_dir / f"{stem}.geff"
        control_hashes[stem] = artifact_tree_sha256(control_path)
        truth_hashes[stem] = artifact_tree_sha256(truth_path)
        if control_hashes[stem] != CONTROL_TREE_SHA256[stem]:
            raise RuntimeError(f"frozen control graph changed: {stem}")
        if truth_hashes[stem] != TRUTH_TREE_SHA256[stem]:
            raise RuntimeError(f"frozen truth graph changed: {stem}")
        scale, estimated = read_truth_metadata(truth_path)
        truth = load_geff_graph(truth_path, verified)
        control = load_geff_graph(control_path, verified)
        movie = candidate_table.filter(pl.col("dataset") == stem)
        node_rows = movie.filter(pl.col("row_type") == "node").sort("node_id")
        edge_rows = movie.filter(pl.col("row_type") == "edge").sort(
            "source_id", "target_id"
        )
        candidate = rebuilder.build_graph_from_rows(node_rows, edge_rows)
        rows["control"].append(
            score_graph(
                verified,
                control,
                truth,
                stem=stem,
                scale=scale,
                estimated_nodes=estimated,
            )
        )
        rows["candidate"].append(
            score_graph(
                verified,
                candidate,
                truth,
                stem=stem,
                scale=scale,
                estimated_nodes=estimated,
            )
        )
    for row in rows["control"]:
        observed = tuple(int(row["official_counts"][name]) for name in COUNT_NAMES)
        if observed != EXPECTED_CONTROL_COUNTS[row["sample_id"]]:
            raise RuntimeError(f"frozen control score changed: {row['sample_id']}")
    pooled = {role: aggregate(verified, role_rows) for role, role_rows in rows.items()}
    by_movie = [
        {
            "sample_id": control["sample_id"],
            "control": public_row(control),
            "candidate": public_row(candidate),
            "score_delta": float(candidate["score"] - control["score"]),
            "adjusted_edge_delta": float(
                candidate["adjusted_edge_jaccard"]
                - control["adjusted_edge_jaccard"]
            ),
        }
        for control, candidate in zip(
            rows["control"], rows["candidate"], strict=True
        )
    ]
    gate = exact_gate(pooled["control"], pooled["candidate"], by_movie)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "accepted" if gate["passed"] else "rejected",
        "evaluation_kind": EVALUATION_KIND,
        "exact_official_gate_passed": gate["passed"],
        "official_scorer_source_verified": True,
        "scorer_lock_sha256": verified.lock_sha256,
        "organizer_commit": verified.lock.organizer_commit,
        "patch_commit": verified.lock.patch_commit,
        "candidate_validator_sha256": candidate_sha256,
        "control_graph_tree_sha256": control_hashes,
        "truth_graph_tree_sha256": truth_hashes,
        "complete_movie_count": len(EXPECTED_STEMS),
        "embryo_prefixes": ["44b6", "6bba"],
        "control": pooled["control"],
        "candidate": pooled["candidate"],
        "delta": metric_delta(pooled["candidate"], pooled["control"]),
        "by_movie": by_movie,
        "gate": gate,
        "competition_test_labels_read": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "authorized_for_submission": gate["passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
