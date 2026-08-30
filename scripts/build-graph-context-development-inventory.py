#!/usr/bin/env python
"""Attach node-only detection context to the frozen EMA development inventory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from biohub_tracker.graphs import artifact_tree_sha256


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "competition-graph-context-development-inventory-v1"
SOURCE_RUN_ID = "competition-relational-division-development-inventory-v1"
SOURCE_SHA256 = "be8ff10d3355e2918a3480cb30a4de57c39a94edbb854aba5c9447da02ab30ce"
EXPECTED_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)


def _load_context_builder() -> Any:
    path = ROOT / "scripts/build-graph-context-relational-archive.py"
    spec = importlib.util.spec_from_file_location("graph_context_archive_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load graph-context feature contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTEXT = _load_context_builder()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_inventory(source_path: Path, prediction_root: Path) -> dict[str, Any]:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("relational development inventory changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    movies = source.get("movies", [])
    if not (
        source.get("schema_version") == 1
        and source.get("status") == "complete"
        and source.get("run_id") == SOURCE_RUN_ID
        and tuple(movie.get("stem") for movie in movies) == EXPECTED_STEMS
        and source.get("summary", {}).get("rows") == 225
        and source.get("summary", {}).get("inference_geometry_eligible_rows") == 9
        and source.get("summary", {}).get("inference_geometry_eligible_positives") == 3
        and source.get("audit_policy_frozen_before_development_probe") is True
        and source.get("competition_test_data_read") is False
        and source.get("public_leaderboard_used_for_selection") is False
        and source.get("authorized_for_submission") is False
    ):
        raise ValueError("relational development inventory is ineligible")
    enriched_movies = []
    valid_tokens = 0
    for movie in movies:
        stem = str(movie["stem"])
        graph_path = prediction_root / f"{stem}.geff"
        graph_hash = artifact_tree_sha256(graph_path)
        if graph_hash != source["prediction_artifacts"][stem]:
            raise ValueError(f"exact EMA development graph changed: {stem}")
        nodes = CONTEXT.load_detection_nodes(graph_path)
        candidates = []
        for row in movie["candidates"]:
            features, mask = CONTEXT.context_tokens(
                nodes,
                parent_id=int(row["parent_id"]),
                existing_child_id=int(row["existing_child_id"]),
                proposed_child_id=int(row["second_child_id"]),
            )
            valid_tokens += int(mask.sum())
            candidates.append(
                {
                    **row,
                    "graph_context_features": features.astype(np.float16).tolist(),
                    "graph_context_mask": mask.tolist(),
                    "graph_context_token_count": CONTEXT.CONTEXT_TOKEN_COUNT,
                    "graph_context_feature_width": CONTEXT.CONTEXT_FEATURE_WIDTH,
                    "graph_context_edges_read": False,
                    "graph_context_labels_used": False,
                }
            )
        enriched_movies.append({**movie, "candidates": candidates})
    result = {
        **source,
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "source_inventory_sha256": SOURCE_SHA256,
        "movies": enriched_movies,
        "summary": {**source["summary"], "graph_context_valid_tokens": valid_tokens},
        "graph_context_token_count": CONTEXT.CONTEXT_TOKEN_COUNT,
        "graph_context_feature_width": CONTEXT.CONTEXT_FEATURE_WIDTH,
        "graph_context_node_arrays_read": list(CONTEXT.NODE_ARRAY_PATHS),
        "graph_context_edge_arrays_read": [],
        "graph_context_edges_read": False,
        "graph_context_labels_used": False,
        "authorized_for_relational_probe_scoring": False,
        "authorized_for_graph_context_probe_scoring": True,
        "authorized_for_submission": False,
    }
    rows = [row for movie in enriched_movies for row in movie["candidates"]]
    if not (
        len(rows) == 225
        and sum(bool(row["inference_geometry_eligible"]) for row in rows) == 9
        and sum(
            bool(row["inference_geometry_eligible"])
            and bool(row["safe_recovery_positive"])
            for row in rows
        )
        == 3
        and valid_tokens >= 225 * 3
    ):
        raise RuntimeError("graph-context development inventory counts changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_inventory(args.source, args.prediction_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
