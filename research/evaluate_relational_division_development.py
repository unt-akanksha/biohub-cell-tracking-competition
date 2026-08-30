#!/usr/bin/env python
"""Evaluate relational/morphology agreement on the exact EMA development graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.division_recovery_feasibility import graph_plain
from research.temporal_contrastive.train_real_division_gate import sha256_file
from research.trackastra_graph.train_biohub_graph_transformer import (
    compute_division_confusion,
    compute_edge_confusion,
    match_nodes_bipartite,
)


RUN_ID = "competition-relational-division-development-v1"
PROBE_RUN_ID = "competition-relational-division-development-probe-v1"
MORPHOLOGY_RUN_ID = "competition-real-handcrafted-division-probe-v1"
BASELINE_RUN_ID = "competition-ranked-consensus-development-baseline-v1"
MORPHOLOGY_SHA256 = "729df1cb9484504a50162da1219aa166d22403cdc117f99300d837a0dfe944f1"


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["stem"]), int(row["timepoint"]), int(row["parent_id"])


def select_agreement_rows(
    relational_rows: list[dict[str, Any]], morphology_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    morphology = {row_key(row): row for row in morphology_rows}
    if any(row_key(row) not in morphology for row in relational_rows):
        raise ValueError("relational development rows are missing morphology evidence")
    selected = []
    for stem in sorted({str(row["stem"]) for row in relational_rows}):
        eligible = [row for row in relational_rows if row["stem"] == stem]
        if not eligible:
            continue
        relational_top = max(
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
        if row_key(relational_top) == row_key(morphology_top):
            result = dict(relational_top)
            result["morphology_probability"] = float(
                morphology[row_key(relational_top)]["ensemble_logit"]
            )
            selected.append(result)
    return selected


def find_graph(root: Path, stem: str) -> Path:
    matches = sorted(path for path in root.rglob(f"{stem}.geff") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"expected one development graph for {stem}, saw {matches}")
    return matches[0]


def confusion(values: tuple[int, int, int]) -> dict[str, Any]:
    tp, fp, fn = (int(value) for value in values)
    return {"tp": tp, "fp": fp, "fn": fn, "jaccard": tp / max(tp + fp + fn, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relational-probe", type=Path, required=True)
    parser.add_argument("--morphology-probe", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    relational = json.loads(args.relational_probe.read_text(encoding="utf-8"))
    morphology = json.loads(args.morphology_probe.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if not (
        relational.get("schema_version") == 1
        and relational.get("status") == "development_probe_complete"
        and relational.get("run_id") == PROBE_RUN_ID
        and relational.get("member_count", 0) >= 1
        and relational.get("metrics", {}).get("rows") == 9
        and relational.get("metrics", {}).get("positives") == 3
        and relational.get("absolute_threshold_used") is False
        and relational.get("weights_searched_on_probe") is False
        and relational.get("model_subset_searched_on_probe") is False
        and relational.get("competition_test_data_read") is False
        and relational.get("public_leaderboard_used_for_selection") is False
        and relational.get("submission_created") is False
        and relational.get("authorized_for_relational_development_evaluation") is True
        and relational.get("authorized_for_submission") is False
        and sha256_file(args.morphology_probe) == MORPHOLOGY_SHA256
        and morphology.get("run_id") == MORPHOLOGY_RUN_ID
        and morphology.get("competition_test_data_read") is False
        and morphology.get("public_leaderboard_used_for_selection") is False
        and morphology.get("submission_created") is False
        and morphology.get("authorized_for_submission") is False
        and baseline.get("schema_version") == 1
        and baseline.get("status") == "verified"
        and baseline.get("run_id") == BASELINE_RUN_ID
        and baseline.get("morphology_probe_sha256") == MORPHOLOGY_SHA256
        and baseline.get("competition_test_data_read") is False
        and baseline.get("public_leaderboard_used_for_selection") is False
        and baseline.get("submission_created") is False
        and baseline.get("authorized_for_submission") is False
    ):
        raise ValueError("relational development evidence is ineligible")
    selected = select_agreement_rows(relational["rows"], morphology["rows"])
    selected_by_stem = {str(row["stem"]): row for row in selected}
    if len(selected_by_stem) != len(selected):
        raise RuntimeError("relational development selected more than one edge per movie")
    totals = {
        "edge_before": [0, 0, 0],
        "edge_after": [0, 0, 0],
        "division_before": [0, 0, 0],
        "division_after": [0, 0, 0],
    }
    movies = []
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
            if pair in set(pred_edges) or pair[1] in incoming:
                raise RuntimeError(f"relational development selected unsafe edge: {stem}/{pair}")
            candidate_edges.append(pair)
        values = {
            "edge_before": compute_edge_confusion(pred_edges, truth_edges, pred_to_truth),
            "edge_after": compute_edge_confusion(candidate_edges, truth_edges, pred_to_truth),
            "division_before": compute_division_confusion(pred_nodes, pred_edges, truth_nodes, truth_edges, pred_to_truth, truth_to_pred),
            "division_after": compute_division_confusion(pred_nodes, candidate_edges, truth_nodes, truth_edges, pred_to_truth, truth_to_pred),
        }
        for name, result in values.items():
            for index, value in enumerate(result):
                totals[name][index] += int(value)
        movies.append(
            {
                "stem": stem,
                "edge_added": row is not None,
                "selected_row": row,
                **{name: confusion(value) for name, value in values.items()},
            }
        )
    pooled = {name: confusion(tuple(value)) for name, value in totals.items()}
    tp = sum(bool(row["safe_recovery_positive"]) for row in selected)
    fp = len(selected) - tp
    frozen = baseline["pooled"]
    authorized = bool(
        len(selected) == 3
        and tp == 3
        and fp == 0
        and pooled["edge_after"]["jaccard"] >= frozen["edge_after"]["jaccard"]
        and pooled["division_after"]["tp"] >= frozen["division_after"]["tp"]
        and pooled["division_after"]["fp"] <= frozen["division_after"]["fp"]
    )
    result = {
        "schema_version": 1,
        "status": "development_positive" if authorized else "development_rejected",
        "run_id": RUN_ID,
        "relational_probe_sha256": sha256_file(args.relational_probe),
        "morphology_probe_sha256": sha256_file(args.morphology_probe),
        "baseline_descriptor_sha256": sha256_file(args.baseline),
        "policy": "per movie, identical top inference-eligible candidate under precommitted relational equal-rank and independent morphology rankings",
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "movies": movies,
        "pooled": pooled,
        "frozen_ranked_consensus_pooled": frozen,
        "absolute_threshold_used": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_full_candidate_evaluation": authorized,
        "authorized_for_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "selected": len(selected), "tp": tp, "fp": fp, "pooled": pooled}, indent=2, sort_keys=True))
    if not authorized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
