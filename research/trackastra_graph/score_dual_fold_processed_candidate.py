#!/usr/bin/env python
"""Score one frozen dual-fold processed candidate with the pinned exact metric.

This CPU-only gate is deliberately downstream of candidate materialization.  It
does not tune association parameters, query the public leaderboard, create a
competition artifact, or submit anything.  The control must reproduce the
previously recorded exact score, and the candidate may differ only in edges.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from biohub_tracker.graphs import load_geff_graph
from biohub_tracker.io import atomic_write_json, sha256_file
from biohub_tracker.scorer_lock import verify_scorer_lock
from biohub_tracker.submission_io import _load_pinned_script
from research.lsm_fm_detection.score_public_node_refinement import (
    aggregate,
    metric_delta,
    public_row,
    score_graph,
)
from research.trackastra_graph.dual_fold_processed_acceptance import (
    EXPECTED_STEMS,
    FROZEN_ASSOCIATION_CONFIGURATION,
    configuration_sha256,
)


RUN_ID = "trackastra-dual-fold-processed-exact-v1"
EVALUATION_KIND = "exact_processed_dual_fold_acceptance"
EXPECTED_CONTROL_SCORE = 0.9343483108193262
MAX_PER_MOVIE_SCORE_REGRESSION = 0.002
FLOAT_TOLERANCE = 1e-12
CSV_COLUMNS = (
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
NODE_IDENTITY_COLUMNS = ("dataset", "node_id", "t", "z", "y", "x")


@dataclass(frozen=True)
class ScoreSample:
    sample_id: str
    scale_zyx_um: tuple[float, float, float]
    estimated_number_of_nodes: int


def read_truth_metadata(path: Path, stem: str) -> ScoreSample:
    """Read only scorer metadata from a GEFF root descriptor."""

    payload = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    geff = payload["attributes"]["geff"]
    axes = {axis["name"]: axis for axis in geff["axes"]}
    if not all(axis in axes for axis in ("z", "y", "x")):
        raise ValueError(f"truth GEFF omits a spatial axis: {stem}")
    scale = tuple(float(axes[axis]["scale"]) for axis in ("z", "y", "x"))
    if any(not math.isfinite(value) or value <= 0 for value in scale):
        raise ValueError(f"truth GEFF has an invalid spatial scale: {stem}")
    estimated = int(geff["extra"]["estimated_number_of_nodes"])
    if estimated <= 0:
        raise ValueError(f"truth GEFF has an invalid estimated node count: {stem}")
    return ScoreSample(stem, scale, estimated)


def validate_materialization(
    payload: Mapping[str, Any],
    *,
    control_sha256: str,
    candidate_sha256: str,
) -> None:
    required = {
        "status": "completed",
        "evaluation_kind": "predeclared_processed_candidate_materialization",
        "gpu_count": 2,
        "whole_movie_sharding": True,
        "ground_truth_read": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
        "exact_processed_scoring_performed": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"materialization is not selection-safe: {mismatches}")
    if payload.get("processed_control_sha256") != control_sha256:
        raise ValueError("materialization control CSV hash mismatch")
    if payload.get("processed_candidate_sha256") != candidate_sha256:
        raise ValueError("materialization candidate CSV hash mismatch")
    if control_sha256 == candidate_sha256:
        raise ValueError("processed candidate is an exact control replica")
    expected_configuration_sha256 = configuration_sha256(
        FROZEN_ASSOCIATION_CONFIGURATION
    )
    if payload.get("association_configuration") != FROZEN_ASSOCIATION_CONFIGURATION:
        raise ValueError("materialization association configuration was not frozen")
    if payload.get("association_configuration_sha256") != expected_configuration_sha256:
        raise ValueError("materialization association configuration hash mismatch")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != {"target_44b6", "target_6bba"}:
        raise ValueError("materialization does not bind both reciprocal models")
    for fold, model in models.items():
        if not isinstance(model, dict) or int(model.get("best_step", 0)) <= 0:
            raise ValueError(f"materialization model did not improve initialization: {fold}")
        digest = model.get("model_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"materialization model hash is invalid: {fold}")
    family = payload.get("candidate_family", "trackastra_dual_fold")
    if family not in {"trackastra_dual_fold", "trackastra_appearance_blend"}:
        raise ValueError(f"unsupported processed candidate family: {family}")
    if family == "trackastra_appearance_blend":
        appearance_models = payload.get("appearance_models")
        appearance_blend = payload.get("appearance_blend")
        folds = {"target_44b6", "target_6bba"}
        if not isinstance(appearance_models, dict) or set(appearance_models) != folds:
            raise ValueError("appearance materialization omits a reciprocal model")
        if not isinstance(appearance_blend, dict) or set(appearance_blend) != folds:
            raise ValueError("appearance materialization omits a reciprocal blend")
        for fold in folds:
            model = appearance_models[fold]
            blend = appearance_blend[fold]
            digest = model.get("model_sha256") if isinstance(model, dict) else None
            if not (
                isinstance(model, dict)
                and int(model.get("best_step", 0)) > 0
                and int(model.get("parameter_count", 0)) == 19_221_954
                and model.get("input_channels") == 3
                and model.get("temporal_frame_offsets") == [-1, 0, 1]
                and isinstance(digest, str)
                and len(digest) == 64
                and model.get("checkpoint_weight_source")
                == "optimizer-step exponential moving average"
                and model.get("ema_decay") == 0.997
                and model.get("link_loss_policy")
                == "all-positive supervised contrastive mean-log-probability"
            ):
                raise ValueError(f"appearance model evidence is invalid: {fold}")
            if not (
                isinstance(blend, dict)
                and float(blend.get("appearance_weight", 0.0)) > 0.0
                and "division_weight" in blend
                and math.isfinite(float(blend["division_weight"]))
                and float(blend["division_weight"]) >= 0.0
                and blend.get("ensemble_mode")
                in {"target_only", "reciprocal_mean"}
                and math.isclose(
                    float(blend.get("appearance_temperature", 0.0)),
                    0.10,
                    rel_tol=0.0,
                    abs_tol=FLOAT_TOLERANCE,
                )
            ):
                raise ValueError(f"appearance blend evidence is invalid: {fold}")
        calibration_digest = payload.get("calibration_terminal_sha256")
        if not isinstance(calibration_digest, str) or len(calibration_digest) != 64:
            raise ValueError("appearance calibration terminal hash is invalid")


def load_submission(path: Path) -> pl.DataFrame:
    table = pl.read_csv(path, columns=list(CSV_COLUMNS))
    if table.height == 0:
        raise ValueError(f"submission CSV is empty: {path}")
    observed = set(table.get_column("dataset").unique().to_list())
    if observed != EXPECTED_STEMS:
        raise ValueError(
            f"processed CSV movie coverage mismatch: expected {sorted(EXPECTED_STEMS)}, "
            f"observed {sorted(observed)}"
        )
    invalid_types = set(table.get_column("row_type").unique().to_list()) - {
        "node",
        "edge",
    }
    if invalid_types:
        raise ValueError(f"processed CSV contains invalid row types: {sorted(invalid_types)}")
    return table


def node_identity_rows(table: pl.DataFrame) -> list[tuple[Any, ...]]:
    return table.filter(pl.col("row_type") == "node").select(
        *NODE_IDENTITY_COLUMNS
    ).sort("dataset", "node_id").rows()


def assert_identical_nodes(control: pl.DataFrame, candidate: pl.DataFrame) -> int:
    control_rows = node_identity_rows(control)
    candidate_rows = node_identity_rows(candidate)
    if control_rows != candidate_rows:
        raise RuntimeError(
            "processed candidate changed node ids, timepoints, or integer coordinates"
        )
    if len(control_rows) != len({(row[0], row[1]) for row in control_rows}):
        raise RuntimeError("processed control contains duplicate dataset/node identifiers")
    return len(control_rows)


def edge_identity_rows(table: pl.DataFrame) -> list[tuple[Any, ...]]:
    return table.filter(pl.col("row_type") == "edge").select(
        "dataset", "source_id", "target_id"
    ).sort("dataset", "source_id", "target_id").rows()


def exact_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    by_movie: Sequence[Mapping[str, Any]],
    *,
    edge_sets_differ: bool,
) -> dict[str, Any]:
    control_reproduced = math.isclose(
        float(control["score"]),
        EXPECTED_CONTROL_SCORE,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    )
    pooled_score_improved = float(candidate["score"]) > float(control["score"])
    movie_deltas = {
        str(row["sample_id"]): float(row["score_delta"]) for row in by_movie
    }
    worst_movie_score_delta = min(movie_deltas.values())
    per_movie_floor_passed = (
        worst_movie_score_delta >= -MAX_PER_MOVIE_SCORE_REGRESSION
    )
    recall_deltas = {
        str(row["sample_id"]): float(row["node_recall_delta"])
        for row in by_movie
    }
    node_recall_identical = all(
        abs(delta) <= FLOAT_TOLERANCE for delta in recall_deltas.values()
    ) and math.isclose(
        float(candidate["node_recall_micro"]),
        float(control["node_recall_micro"]),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    )
    checks = {
        "control_score_reproduced": control_reproduced,
        "pooled_score_improved": pooled_score_improved,
        "per_movie_regression_floor_passed": per_movie_floor_passed,
        "node_recall_identical": node_recall_identical,
        "edge_sets_differ": bool(edge_sets_differ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_control_score": EXPECTED_CONTROL_SCORE,
        "maximum_per_movie_score_regression": MAX_PER_MOVIE_SCORE_REGRESSION,
        "worst_movie_score_delta": worst_movie_score_delta,
        "score_delta_by_movie": movie_deltas,
        "node_recall_delta_by_movie": recall_deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--scorer-lock", type=Path, required=True)
    parser.add_argument("--organizer-checkout", type=Path, required=True)
    parser.add_argument("--tracksdata-checkout", type=Path, required=True)
    parser.add_argument("--materialization-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite exact evidence: {args.output}")

    control_sha256 = sha256_file(args.control_csv)
    candidate_sha256 = sha256_file(args.candidate_csv)
    materialization = json.loads(
        args.materialization_result.read_text(encoding="utf-8")
    )
    validate_materialization(
        materialization,
        control_sha256=control_sha256,
        candidate_sha256=candidate_sha256,
    )

    control_table = load_submission(args.control_csv)
    candidate_table = load_submission(args.candidate_csv)
    node_count = assert_identical_nodes(control_table, candidate_table)
    edge_sets_differ = edge_identity_rows(control_table) != edge_identity_rows(
        candidate_table
    )
    if not edge_sets_differ:
        raise RuntimeError("processed candidate has the exact control edge set")

    verified = verify_scorer_lock(
        args.scorer_lock,
        args.organizer_checkout,
        tracksdata_checkout=args.tracksdata_checkout,
    )
    rebuilder = _load_pinned_script(verified, "csv_to_geffs.py")
    rows: dict[str, list[dict[str, Any]]] = {"control": [], "candidate": []}
    truth_sha256: dict[str, str] = {}
    for stem in sorted(EXPECTED_STEMS):
        truth_path = args.truth_dir / f"{stem}.geff"
        if not truth_path.is_dir():
            raise FileNotFoundError(truth_path)
        sample = read_truth_metadata(truth_path, stem)
        truth = load_geff_graph(truth_path, verified)
        truth_sha256[stem] = sha256_file(truth_path / "zarr.json")
        for role, table in (("control", control_table), ("candidate", candidate_table)):
            movie = table.filter(pl.col("dataset") == stem)
            node_rows = movie.filter(pl.col("row_type") == "node").sort("node_id")
            edge_rows = movie.filter(pl.col("row_type") == "edge").sort(
                "source_id", "target_id"
            )
            graph = rebuilder.build_graph_from_rows(node_rows, edge_rows)
            rows[role].append(score_graph(verified, graph, truth, sample))

    pooled = {role: aggregate(verified, role_rows) for role, role_rows in rows.items()}
    by_movie = [
        {
            "sample_id": base["sample_id"],
            "control": public_row(base),
            "candidate": public_row(challenger),
            "score_delta": float(challenger["score"] - base["score"]),
            "node_recall_delta": float(
                challenger["node_recall"] - base["node_recall"]
            ),
        }
        for base, challenger in zip(rows["control"], rows["candidate"], strict=True)
    ]
    gate = exact_gate(
        pooled["control"],
        pooled["candidate"],
        by_movie,
        edge_sets_differ=edge_sets_differ,
    )
    status = "accepted" if gate["passed"] else "rejected"
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": status,
        "evaluation_kind": EVALUATION_KIND,
        "exact_processed_gate_passed": gate["passed"],
        "authorized_for_submission": False,
        "competition_submission_performed": False,
        "public_leaderboard_used_for_selection": False,
        "hyperparameter_selection_performed": False,
        "candidate_node_rows_identical": True,
        "candidate_edge_sets_differ": edge_sets_differ,
        "processed_node_count": node_count,
        "processed_control_sha256": control_sha256,
        "processed_candidate_sha256": candidate_sha256,
        "materialization_result_sha256": sha256_file(args.materialization_result),
        "scorer_lock_sha256": verified.lock_sha256,
        "truth_root_descriptor_sha256": truth_sha256,
        "association_configuration": FROZEN_ASSOCIATION_CONFIGURATION,
        "association_configuration_sha256": configuration_sha256(
            FROZEN_ASSOCIATION_CONFIGURATION
        ),
        "models": materialization["models"],
        "pooled": pooled,
        "delta": metric_delta(pooled["candidate"], pooled["control"]),
        "by_movie": by_movie,
        "gate": gate,
        "note": "Exact CPU acceptance only; no leaderboard query, artifact promotion, or submission was performed.",
    }
    if materialization.get("candidate_family") == "trackastra_appearance_blend":
        result.update(
            {
                "candidate_family": "trackastra_appearance_blend",
                "appearance_models": materialization["appearance_models"],
                "appearance_blend": materialization["appearance_blend"],
                "calibration_terminal_sha256": materialization[
                    "calibration_terminal_sha256"
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
