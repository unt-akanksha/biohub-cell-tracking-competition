#!/usr/bin/env python
"""Exact diagnostic scoring for public-node coordinate refinement.

This scorer intentionally has no promotion or submission path. It verifies the
pinned organizer implementation, compares the frozen public graph with the
refined graph, and scores both native coordinates and the integer coordinates
that the official GEFF-to-CSV converter would actually submit.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from biohub_tracker.graphs import (
    artifact_tree_sha256,
    graph_data_from_tracksdata,
    load_geff_graph,
)
from biohub_tracker.io import atomic_write_json, sha256_file
from biohub_tracker.manifests import SampleRecord, load_manifest
from biohub_tracker.scorer_lock import verify_scorer_lock
from biohub_tracker.submission_io import _load_pinned_script


STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)


def finite_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def graph_topology(graph: Any) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    data = graph_data_from_tracksdata(graph)
    nodes = tuple(sorted((int(node.node_id), int(node.t)) for node in data.nodes))
    edges = tuple(sorted((int(source), int(target)) for source, target in data.edges))
    return nodes, edges


def assert_same_topology(control: Any, candidate: Any, stem: str) -> None:
    if graph_topology(control) != graph_topology(candidate):
        raise RuntimeError(f"public-node refinement changed topology: {stem}")


def submission_graph(converter: Any, rebuilder: Any, graph: Any, stem: str) -> tuple[Any, Any]:
    import polars as pl

    rows = converter.graph_to_rows(graph, stem)
    node_rows = rows.filter(pl.col("row_type") == "node").sort("node_id")
    edge_rows = rows.filter(pl.col("row_type") == "edge").sort(
        "source_id", "target_id"
    )
    return rebuilder.build_graph_from_rows(node_rows, edge_rows), node_rows


def changed_rounded_nodes(control_rows: Any, candidate_rows: Any) -> int:
    axes = ("z", "y", "x")
    control = {
        int(row["node_id"]): tuple(int(row[axis]) for axis in axes)
        for row in control_rows.select("node_id", *axes).iter_rows(named=True)
    }
    candidate = {
        int(row["node_id"]): tuple(int(row[axis]) for axis in axes)
        for row in candidate_rows.select("node_id", *axes).iter_rows(named=True)
    }
    if control.keys() != candidate.keys():
        raise RuntimeError("integer projection changed node identifiers")
    return sum(control[node_id] != candidate[node_id] for node_id in control)


def score_graph(verified: Any, prediction: Any, truth: Any, sample: SampleRecord) -> dict[str, Any]:
    scored_prediction = prediction.copy()
    scored_truth = truth.copy()
    result = verified.evaluate(
        scored_prediction,
        scored_truth,
        scale=tuple(float(value) for value in sample.scale_zyx_um),
        max_distance=float(verified.lock.raw["constants"]["max_distance_um"]),
    )
    recall = float(verified.node_recall(scored_prediction, scored_truth))
    row = verified.per_sample_metrics(
        result, float(sample.estimated_number_of_nodes), recall
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No divisions present across any sample in this split*"
        )
        summary = verified.summarise([row])
    counts = {
        name: int(getattr(result, name))
        for name in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_tp",
            "division_fp",
            "division_fn",
            "num_pred_nodes",
        )
    }
    return {
        "sample_id": sample.sample_id,
        "organizer_row": dict(row),
        "official_counts": counts,
        "gt_node_count": int(scored_truth.num_nodes()),
        "matched_gt_node_count": round(recall * scored_truth.num_nodes()),
        "estimated_number_of_nodes": sample.estimated_number_of_nodes,
        "node_recall": recall,
        "edge_jaccard": float(row["edge_jaccard"]),
        "adjusted_edge_jaccard": float(row["adj_edge_jaccard"]),
        "division_jaccard": finite_float(summary["division_jaccard"]),
        "score": float(summary["score"]),
    }


def aggregate(verified: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    organizer_rows = [dict(row["organizer_row"]) for row in rows]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No divisions present across any sample in this split*"
        )
        summary = verified.summarise(organizer_rows)
    counts = {
        name: sum(int(row["official_counts"][name]) for row in rows)
        for name in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "division_tp",
            "division_fp",
            "division_fn",
            "num_pred_nodes",
        )
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
    names = (
        "score",
        "adjusted_edge_jaccard",
        "edge_jaccard",
        "division_jaccard",
        "node_recall_micro",
    )
    result: dict[str, float | None] = {}
    for name in names:
        left, right = candidate.get(name), control.get(name)
        result[name] = None if left is None or right is None else float(left) - float(right)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--scorer-lock", type=Path, required=True)
    parser.add_argument("--organizer-checkout", type=Path, required=True)
    parser.add_argument("--tracksdata-checkout", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic evidence: {args.output}")
    source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
    if not (
        source_result.get("status") == "completed"
        and source_result.get("selection_passed") is True
        and source_result.get("acceptance_opened") is True
        and source_result.get("public_graph_used_as_base") is True
        and source_result.get("exact_public_replica") is False
        and source_result.get("competition_submission_performed") is False
    ):
        raise ValueError("public-node source result is not eligible for exact diagnostics")

    manifest = load_manifest(args.manifest)
    samples = {sample.sample_id: sample for sample in manifest.samples}
    if any(stem not in samples for stem in STEMS):
        raise ValueError("evaluation manifest omits a public-node acceptance movie")
    verified = verify_scorer_lock(
        args.scorer_lock,
        args.organizer_checkout,
        tracksdata_checkout=args.tracksdata_checkout,
    )
    converter = _load_pinned_script(verified, "geffs_to_csv.py")
    rebuilder = _load_pinned_script(verified, "csv_to_geffs.py")

    scored: dict[str, dict[str, list[dict[str, Any]]]] = {
        space: {role: [] for role in ("control", "candidate")}
        for space in ("native", "integer_submission")
    }
    topology: dict[str, Any] = {}
    for stem in STEMS:
        sample = samples[stem]
        paths = {
            "truth": args.truth_dir / f"{stem}.geff",
            "control": args.control_dir / f"{stem}.geff",
            "candidate": args.candidate_dir / f"{stem}.geff",
        }
        if any(not path.is_dir() for path in paths.values()):
            raise FileNotFoundError({name: str(path) for name, path in paths.items()})
        graphs = {name: load_geff_graph(path, verified) for name, path in paths.items()}
        assert_same_topology(graphs["control"], graphs["candidate"], stem)
        control_submission, control_rows = submission_graph(
            converter, rebuilder, graphs["control"], stem
        )
        candidate_submission, candidate_rows = submission_graph(
            converter, rebuilder, graphs["candidate"], stem
        )
        topology[stem] = {
            "topology_preserved": True,
            "nodes": int(graphs["candidate"].num_nodes()),
            "edges": int(graphs["candidate"].num_edges()),
            "rounded_coordinate_nodes_changed": changed_rounded_nodes(
                control_rows, candidate_rows
            ),
            "control_graph_sha256": artifact_tree_sha256(paths["control"]),
            "candidate_graph_sha256": artifact_tree_sha256(paths["candidate"]),
            "truth_graph_sha256": artifact_tree_sha256(paths["truth"]),
        }
        for role, graph in (
            ("control", graphs["control"]),
            ("candidate", graphs["candidate"]),
        ):
            scored["native"][role].append(
                score_graph(verified, graph, graphs["truth"], sample)
            )
        for role, graph in (
            ("control", control_submission),
            ("candidate", candidate_submission),
        ):
            scored["integer_submission"][role].append(
                score_graph(verified, graph, graphs["truth"], sample)
            )

    comparisons: dict[str, Any] = {}
    for space, roles in scored.items():
        control = aggregate(verified, roles["control"])
        candidate = aggregate(verified, roles["candidate"])
        comparisons[space] = {
            "control": control,
            "candidate": candidate,
            "delta": metric_delta(candidate, control),
            "by_movie": [
                {
                    "sample_id": base["sample_id"],
                    "control": public_row(base),
                    "candidate": public_row(challenger),
                    "score_delta": challenger["score"] - base["score"],
                    "node_recall_delta": challenger["node_recall"] - base["node_recall"],
                }
                for base, challenger in zip(
                    roles["control"], roles["candidate"], strict=True
                )
            ],
        }
    authoritative_delta = comparisons["integer_submission"]["delta"]
    result = {
        "schema_version": 1,
        "status": "completed",
        "evidence_kind": "diagnostic_only_exact_metric",
        "authorized_for_promotion": False,
        "authorized_for_submission": False,
        "competition_submission_performed": False,
        "public_graph_used_as_base": True,
        "exact_public_replica": False,
        "official_scorer_source_verified": True,
        "scorer_lock_sha256": verified.lock_sha256,
        "source_result_sha256": sha256_file(args.source_result),
        "topology": topology,
        "comparisons": comparisons,
        "integer_submission_score_improved": (
            authoritative_delta["score"] is not None
            and authoritative_delta["score"] > 0.0
        ),
        "integer_submission_node_recall_nonregressed": (
            authoritative_delta["node_recall_micro"] is not None
            and authoritative_delta["node_recall_micro"] >= 0.0
        ),
        "note": "This exact diagnostic is selection-safe evidence, not ledger-bound promotion authority.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
