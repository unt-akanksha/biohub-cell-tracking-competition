#!/usr/bin/env python
"""Evaluate scale-invariant deep/morphology consensus on development graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.division_recovery_feasibility import graph_plain
from research.trackastra_graph.train_biohub_graph_transformer import (
    compute_division_confusion,
    compute_edge_confusion,
    match_nodes_bipartite,
)


RUN_ID = "competition-ranked-consensus-division-development-v1"
GEOMETRY_MINIMUM = 3.0


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["stem"]), int(row["timepoint"]), int(row["parent_id"])


def select_consensus_rows(
    deep_rows: list[dict[str, Any]], morphology_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    morphology = {row_key(row): row for row in morphology_rows}
    if set(morphology) != {row_key(row) for row in deep_rows}:
        raise ValueError("deep and morphology probe inventories disagree")
    selected = []
    stems = sorted({str(row["stem"]) for row in deep_rows})
    for stem in stems:
        eligible = [
            row
            for row in deep_rows
            if row["stem"] == stem
            and float(row["geometry_division_score"]) >= GEOMETRY_MINIMUM
        ]
        if not eligible:
            continue
        deep_top = max(
            eligible,
            key=lambda row: (
                float(row["ensemble_logit"]),
                -int(row["timepoint"]),
                -int(row["parent_id"]),
            ),
        )
        morphology_top = max(
            eligible,
            key=lambda row: (
                float(morphology[row_key(row)]["ensemble_logit"]),
                -int(row["timepoint"]),
                -int(row["parent_id"]),
            ),
        )
        if row_key(deep_top) != row_key(morphology_top):
            continue
        result = dict(deep_top)
        result["morphology_probability"] = float(
            morphology[row_key(deep_top)]["ensemble_logit"]
        )
        selected.append(result)
    return selected


def find_graph(root: Path, stem: str) -> Path:
    matches = sorted(path for path in root.rglob(f"{stem}.geff") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"expected one graph for {stem}, saw {matches}")
    return matches[0]


def confusion_dict(values: tuple[int, int, int]) -> dict[str, Any]:
    tp, fp, fn = values
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "jaccard": float(tp / max(tp + fp + fn, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-probe", type=Path, required=True)
    parser.add_argument("--morphology-probe", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deep = json.loads(args.deep_probe.read_text(encoding="utf-8"))
    morphology = json.loads(args.morphology_probe.read_text(encoding="utf-8"))
    if not (
        deep.get("run_id") == "competition-train-focused-division-transfer-probe-v1"
        and deep.get("competition_test_data_read") is False
        and morphology.get("run_id") == "competition-real-handcrafted-division-probe-v1"
        and morphology.get("competition_test_data_read") is False
    ):
        raise ValueError("ranked consensus probes are ineligible")
    selected = select_consensus_rows(deep["rows"], morphology["rows"])
    selected_by_stem = {row["stem"]: row for row in selected}
    before_edge = [0, 0, 0]
    after_edge = [0, 0, 0]
    before_division = [0, 0, 0]
    after_division = [0, 0, 0]
    movie_results = []
    for truth_path in sorted(args.truth_root.glob("*.geff")):
        stem = truth_path.stem
        prediction_path = find_graph(args.prediction_root, stem)
        pred_nodes, pred_edges = graph_plain(prediction_path)
        truth_nodes, truth_edges = graph_plain(truth_path)
        pred_to_truth, truth_to_pred = match_nodes_bipartite(
            pred_nodes, truth_nodes, max_dist=7.0
        )
        candidate_edges = list(pred_edges)
        row = selected_by_stem.get(stem)
        if row is not None:
            pair = (int(row["parent_id"]), int(row["second_child_id"]))
            incoming = {target for _source, target in pred_edges}
            if (
                pair[0] not in pred_nodes
                or pair[1] not in pred_nodes
                or pair in set(pred_edges)
                or pair[1] in incoming
            ):
                raise RuntimeError(f"ranked consensus selected an unsafe edge: {stem}/{pair}")
            candidate_edges.append(pair)
        base_edge = compute_edge_confusion(pred_edges, truth_edges, pred_to_truth)
        candidate_edge = compute_edge_confusion(
            candidate_edges, truth_edges, pred_to_truth
        )
        base_division = compute_division_confusion(
            pred_nodes,
            pred_edges,
            truth_nodes,
            truth_edges,
            pred_to_truth,
            truth_to_pred,
        )
        candidate_division = compute_division_confusion(
            pred_nodes,
            candidate_edges,
            truth_nodes,
            truth_edges,
            pred_to_truth,
            truth_to_pred,
        )
        for total, values in (
            (before_edge, base_edge),
            (after_edge, candidate_edge),
            (before_division, base_division),
            (after_division, candidate_division),
        ):
            for index, value in enumerate(values):
                total[index] += int(value)
        movie_results.append(
            {
                "stem": stem,
                "edge_added": row is not None,
                "selected_row": row,
                "edge_before": confusion_dict(base_edge),
                "edge_after": confusion_dict(candidate_edge),
                "division_before": confusion_dict(base_division),
                "division_after": confusion_dict(candidate_division),
            }
        )
    before_edge_result = confusion_dict(tuple(before_edge))
    after_edge_result = confusion_dict(tuple(after_edge))
    before_division_result = confusion_dict(tuple(before_division))
    after_division_result = confusion_dict(tuple(after_division))
    tp = sum(bool(row["safe_recovery_positive"]) for row in selected)
    fp = len(selected) - tp
    authorized = bool(
        len(selected) >= 1
        and tp == len(selected)
        and fp == 0
        and after_edge_result["jaccard"] > before_edge_result["jaccard"]
        and after_division_result["tp"] > before_division_result["tp"]
        and after_division_result["fp"] <= before_division_result["fp"]
    )
    result = {
        "schema_version": 1,
        "status": "development_positive" if authorized else "development_rejected",
        "run_id": RUN_ID,
        "policy": (
            "per movie, require geometry >=3 and identical top parent under the "
            "deep and independent morphology rankings; add at most one parent-free edge"
        ),
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "movies": movie_results,
        "pooled": {
            "edge_before": before_edge_result,
            "edge_after": after_edge_result,
            "division_before": before_division_result,
            "division_after": after_division_result,
        },
        "absolute_threshold_used": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_full_candidate_evaluation": authorized,
        "authorized_for_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["pooled"], indent=2, sort_keys=True))
    if not authorized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
