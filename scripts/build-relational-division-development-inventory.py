#!/usr/bin/env python
"""Bind exact EMA development candidates to all three relational centers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub_tracker.graphs import artifact_tree_sha256
from research.division_recovery_feasibility import graph_plain
from research.learned_division_recovery import discover_division_recovery_candidates


RUN_ID = "competition-relational-division-development-inventory-v1"
SOURCE_INVENTORY_SHA256 = "8bd7e47ffadcb1429c90b3d9a3757fc04ae2ce849fb5a3d49d0c7f7dceb46127"
BASELINE_RUN_ID = "competition-ranked-consensus-development-baseline-v1"
EXPECTED_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
NUMERIC_FIELDS = (
    "parent_distance_um",
    "sister_distance_um",
    "existing_distance_um",
    "daughter_midpoint_distance_um",
    "daughter_opposition_cosine",
    "daughter_step_ratio",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def node_mapping(nodes: dict[int, tuple[float, ...]]) -> dict[int, dict[str, float | int]]:
    return {
        int(node_id): {
            "t": int(row[0]),
            "z": float(row[1]),
            "y": float(row[2]),
            "x": float(row[3]),
        }
        for node_id, row in nodes.items()
    }


def candidate_key(row: Any) -> tuple[int, int, int]:
    if isinstance(row, dict):
        return (
            int(row["parent_id"]),
            int(row["existing_child_id"]),
            int(row["second_child_id"]),
        )
    return (int(row.parent_id), int(row.existing_child_id), int(row.second_child_id))


def build_inventory(
    *,
    source_path: Path,
    prediction_root: Path,
    baseline_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_path) != SOURCE_INVENTORY_SHA256:
        raise ValueError("development division probe inventory changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source_movies = {str(row["stem"]): row for row in source.get("movies", [])}
    if not (
        source.get("schema_version") == 1
        and source.get("status") == "competition_train_probe_only"
        and source.get("competition_train_data_read") is True
        and source.get("competition_test_data_read") is False
        and source.get("public_leaderboard_used_for_selection") is False
        and source.get("submission_created") is False
        and source.get("authorized_for_submission") is False
        and tuple(sorted(source_movies)) == EXPECTED_STEMS
        and baseline.get("schema_version") == 1
        and baseline.get("status") == "verified"
        and baseline.get("run_id") == BASELINE_RUN_ID
        and tuple(baseline.get("stems", [])) == EXPECTED_STEMS
        and baseline.get("competition_test_data_read") is False
        and baseline.get("public_leaderboard_used_for_selection") is False
        and baseline.get("submission_created") is False
        and baseline.get("authorized_for_submission") is False
    ):
        raise ValueError("relational development sources are ineligible")
    movies = []
    graph_artifacts = {}
    for stem in EXPECTED_STEMS:
        graph_path = prediction_root / f"{stem}.geff"
        graph_hash = artifact_tree_sha256(graph_path)
        if graph_hash != baseline["prediction_artifacts"][stem]:
            raise ValueError(f"exact EMA development graph changed: {stem}")
        graph_artifacts[stem] = graph_hash
        plain_nodes, plain_edges = graph_plain(graph_path)
        nodes = node_mapping(plain_nodes)
        edges = [
            {"source_id": int(source_id), "target_id": int(target_id)}
            for source_id, target_id in plain_edges
        ]
        discovered = discover_division_recovery_candidates(nodes, edges)
        by_key = {candidate_key(row): row for row in discovered}
        source_movie = source_movies[stem]
        source_rows = source_movie.get("candidates", [])
        source_keys = {candidate_key(row) for row in source_rows}
        if not source_keys <= set(by_key):
            missing = sorted(source_keys - set(by_key))
            raise ValueError(
                f"EMA relational candidate inventory changed: {stem}; "
                f"source={len(source_keys)} ema={len(by_key)} "
                f"missing={missing[:5]}"
            )
        enriched = []
        for row in source_rows:
            key = candidate_key(row)
            candidate = by_key[key]
            if any(
                not math.isclose(
                    float(row[name]), float(getattr(candidate, name)), rel_tol=0.0, abs_tol=1e-9
                )
                for name in NUMERIC_FIELDS
            ):
                raise ValueError(f"EMA relational candidate geometry changed: {stem}/{key}")
            centers = {
                name: [
                    float(nodes[identifier][axis]) for axis in ("z", "y", "x")
                ]
                for name, identifier in (
                    ("parent", candidate.parent_id),
                    ("existing_child", candidate.existing_child_id),
                    ("proposed_child", candidate.second_child_id),
                )
            }
            if centers["parent"] != [float(value) for value in row["center_zyx_voxel"]]:
                raise ValueError(f"EMA parent center changed: {stem}/{key}")
            enriched.append(
                {
                    **row,
                    "biological_geometry_score": float(candidate.biological_geometry_score),
                    "centers_zyx_voxel": centers,
                    "inference_geometry_eligible": bool(
                        candidate.biological_geometry_score >= 3.0
                    ),
                    "daughter_order_invariant": True,
                }
            )
        movies.append(
            {
                **source_movie,
                "candidates": enriched,
                "ema_graph_sha256": graph_hash,
                "ema_all_timepoint_candidate_count": len(by_key),
                "development_scoped_candidate_count": len(enriched),
            }
        )
    rows = [row for movie in movies for row in movie["candidates"]]
    eligible = [row for row in rows if row["inference_geometry_eligible"]]
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "source_inventory_sha256": sha256_file(source_path),
        "baseline_descriptor_sha256": sha256_file(baseline_path),
        "prediction_artifacts": graph_artifacts,
        "movies": movies,
        "summary": {
            "movies": len(movies),
            "rows": len(rows),
            "safe_recovery_positives": sum(
                bool(row["safe_recovery_positive"]) for row in rows
            ),
            "inference_geometry_eligible_rows": len(eligible),
            "inference_geometry_eligible_positives": sum(
                bool(row["safe_recovery_positive"]) for row in eligible
            ),
            "ema_all_timepoint_candidates": sum(
                int(movie["ema_all_timepoint_candidate_count"]) for movie in movies
            ),
        },
        "audit_policy_frozen_before_development_probe": True,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "final_development_probe_opened": True,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_relational_probe_scoring": True,
        "authorized_for_submission": False,
    }
    if not (
        result["summary"]["movies"] == 4
        and result["summary"]["rows"] == 225
        and result["summary"]["safe_recovery_positives"] == 3
        and result["summary"]["inference_geometry_eligible_positives"] == 3
        and result["summary"]["inference_geometry_eligible_rows"] > 3
    ):
        raise RuntimeError("relational development inventory counts changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_inventory(
        source_path=args.source_inventory,
        prediction_root=args.prediction_root,
        baseline_path=args.baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
