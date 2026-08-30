#!/usr/bin/env python
"""Evaluate graph-context/morphology agreement on frozen EMA development graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.division_recovery_feasibility import graph_plain
from research.evaluate_relational_division_development import (
    confusion,
    find_graph,
    select_agreement_rows,
)
from research.temporal_contrastive.train_real_division_gate import sha256_file
from research.trackastra_graph.train_biohub_graph_transformer import (
    compute_division_confusion,
    compute_edge_confusion,
    match_nodes_bipartite,
)


RUN_ID = "competition-graph-context-division-development-v1"
PROBE_RUN_ID = "competition-graph-context-division-development-probe-v1"
MORPHOLOGY_RUN_ID = "competition-real-handcrafted-division-probe-v1"
BASELINE_RUN_ID = "competition-ranked-consensus-development-baseline-v1"
MORPHOLOGY_SHA256 = "729df1cb9484504a50162da1219aa166d22403cdc117f99300d837a0dfe944f1"


def evaluate(
    *,
    probe_path: Path,
    morphology_path: Path,
    prediction_root: Path,
    truth_root: Path,
    baseline_path: Path,
) -> dict[str, Any]:
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    morphology = json.loads(morphology_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not (
        probe.get("schema_version") == 1
        and probe.get("status") == "development_probe_complete"
        and probe.get("run_id") == PROBE_RUN_ID
        and probe.get("member_count", 0) >= 1
        and probe.get("metrics", {}).get("rows") == 9
        and probe.get("metrics", {}).get("positives") == 3
        and probe.get("absolute_threshold_used") is False
        and probe.get("weights_searched_on_probe") is False
        and probe.get("model_subset_searched_on_probe") is False
        and probe.get("competition_test_data_read") is False
        and probe.get("public_leaderboard_used_for_selection") is False
        and probe.get("submission_created") is False
        and probe.get("authorized_for_graph_context_development_evaluation") is True
        and probe.get("authorized_for_submission") is False
        and sha256_file(morphology_path) == MORPHOLOGY_SHA256
        and morphology.get("run_id") == MORPHOLOGY_RUN_ID
        and morphology.get("competition_test_data_read") is False
        and morphology.get("public_leaderboard_used_for_selection") is False
        and morphology.get("authorized_for_submission") is False
        and baseline.get("schema_version") == 1
        and baseline.get("status") == "verified"
        and baseline.get("run_id") == BASELINE_RUN_ID
        and baseline.get("morphology_probe_sha256") == MORPHOLOGY_SHA256
        and baseline.get("competition_test_data_read") is False
        and baseline.get("public_leaderboard_used_for_selection") is False
        and baseline.get("authorized_for_submission") is False
    ):
        raise ValueError("graph-context development evidence is ineligible")
    selected = select_agreement_rows(probe["rows"], morphology["rows"])
    selected_by_stem = {str(row["stem"]): row for row in selected}
    if len(selected_by_stem) != len(selected):
        raise RuntimeError("graph-context development selected multiple edges per movie")
    totals = {
        "edge_before": [0, 0, 0],
        "edge_after": [0, 0, 0],
        "division_before": [0, 0, 0],
        "division_after": [0, 0, 0],
    }
    movies = []
    for truth_path in sorted(truth_root.glob("*.geff")):
        stem = truth_path.stem
        prediction_path = find_graph(prediction_root, stem)
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
                raise RuntimeError(
                    f"graph-context development selected unsafe edge: {stem}/{pair}"
                )
            candidate_edges.append(pair)
        values = {
            "edge_before": compute_edge_confusion(pred_edges, truth_edges, pred_to_truth),
            "edge_after": compute_edge_confusion(candidate_edges, truth_edges, pred_to_truth),
            "division_before": compute_division_confusion(
                pred_nodes,
                pred_edges,
                truth_nodes,
                truth_edges,
                pred_to_truth,
                truth_to_pred,
            ),
            "division_after": compute_division_confusion(
                pred_nodes,
                candidate_edges,
                truth_nodes,
                truth_edges,
                pred_to_truth,
                truth_to_pred,
            ),
        }
        for name, values_row in values.items():
            for index, value in enumerate(values_row):
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
    true_positives = sum(bool(row["safe_recovery_positive"]) for row in selected)
    false_positives = len(selected) - true_positives
    frozen = baseline["pooled"]
    authorized = bool(
        len(selected) == 3
        and true_positives == 3
        and false_positives == 0
        and pooled["edge_after"]["jaccard"] >= frozen["edge_after"]["jaccard"]
        and pooled["division_after"]["tp"] >= frozen["division_after"]["tp"]
        and pooled["division_after"]["fp"] <= frozen["division_after"]["fp"]
    )
    return {
        "schema_version": 1,
        "status": "development_positive" if authorized else "development_rejected",
        "run_id": RUN_ID,
        "graph_context_probe_sha256": sha256_file(probe_path),
        "morphology_probe_sha256": sha256_file(morphology_path),
        "baseline_descriptor_sha256": sha256_file(baseline_path),
        "policy": "per movie, identical top inference-eligible candidate under precommitted graph-context equal-rank and independent morphology rankings",
        "selected": len(selected),
        "tp": true_positives,
        "fp": false_positives,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-context-probe", type=Path, required=True)
    parser.add_argument("--morphology-probe", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        probe_path=args.graph_context_probe,
        morphology_path=args.morphology_probe,
        prediction_root=args.prediction_root,
        truth_root=args.truth_root,
        baseline_path=args.baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": result["selected"],
                "tp": result["tp"],
                "fp": result["fp"],
                "pooled": result["pooled"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["status"] != "development_positive":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
