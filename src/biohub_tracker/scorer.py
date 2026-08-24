from __future__ import annotations

import json
import math
import secrets
from decimal import Decimal
from pathlib import Path
from typing import Any

from .io import canonical_json_bytes, sha256_bytes
from .scorer_lock import ScorerVerificationError, VerifiedScorer


_COUNT_FIELDS = (
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "division_tp",
    "division_fp",
    "division_fn",
    "num_pred_nodes",
)
_SUMMARY_FIELDS = (
    "n",
    "edge_jaccard",
    "division_jaccard",
    "division_tp",
    "division_fp",
    "division_fn",
    "node_recall",
    "adj_edge_jaccard",
    "n_adj",
    "score",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScorerVerificationError("fixture_unreadable", str(exc)) from exc
    if not isinstance(value, dict):
        raise ScorerVerificationError("fixture_schema_invalid", "fixture root must be an object")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScorerVerificationError("fixture_schema_invalid", f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScorerVerificationError("fixture_schema_invalid", f"{name} must be finite")
    return result


def _graph_from_spec(verified: VerifiedScorer, value: Any):
    if not isinstance(value, dict) or set(value) != {"nodes", "edges"}:
        raise ScorerVerificationError("fixture_schema_invalid", "graph keys must be nodes/edges")
    nodes = value["nodes"]
    edges = value["edges"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 10_000:
        raise ScorerVerificationError("fixture_schema_invalid", "invalid node count")
    if not isinstance(edges, list) or len(edges) > 20_000:
        raise ScorerVerificationError("fixture_schema_invalid", "invalid edge count")

    polars = __import__("polars")
    graph = verified.tracksdata.graph.InMemoryGraph()
    for coordinate in ("z", "y", "x"):
        graph.add_node_attr_key(coordinate, polars.Float64, 0.0)
    ids: dict[str, int] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != {"id", "t", "z", "y", "x"}:
            raise ScorerVerificationError("fixture_schema_invalid", f"node {index} has invalid keys")
        name = node["id"]
        time = node["t"]
        if not isinstance(name, str) or not name or name in ids:
            raise ScorerVerificationError("fixture_schema_invalid", f"node {index} has invalid id")
        if isinstance(time, bool) or not isinstance(time, int):
            raise ScorerVerificationError("fixture_schema_invalid", f"node {name} has invalid time")
        ids[name] = graph.add_node(
            {
                "t": time,
                "z": _finite_number(node["z"], f"{name}.z"),
                "y": _finite_number(node["y"], f"{name}.y"),
                "x": _finite_number(node["x"], f"{name}.x"),
            }
        )
    seen_edges: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2 or not all(isinstance(item, str) for item in edge):
            raise ScorerVerificationError("fixture_schema_invalid", f"edge {index} is invalid")
        pair = (edge[0], edge[1])
        if pair in seen_edges or pair[0] not in ids or pair[1] not in ids:
            raise ScorerVerificationError("fixture_schema_invalid", f"edge {index} is invalid")
        seen_edges.add(pair)
        graph.add_edge(ids[pair[0]], ids[pair[1]], {})
    return graph


def _decimal_text(value: float) -> str:
    if not math.isfinite(value):
        raise ScorerVerificationError("non_finite_metric", str(value))
    decimal = Decimal(str(value))
    if decimal == 0:
        return "0"
    return format(decimal.normalize(), "f")


def _canonical_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _SUMMARY_FIELDS:
        if name not in summary:
            raise ScorerVerificationError("official_summary_incomplete", name)
        value = summary[name]
        if name in {"n", "n_adj", "division_tp", "division_fp", "division_fn"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ScorerVerificationError("official_summary_invalid", name)
            result[name] = value
        elif isinstance(value, (int, float)) and math.isnan(float(value)):
            if name != "division_jaccard":
                raise ScorerVerificationError("non_finite_metric", name)
            result[name] = None
        else:
            result[name] = _decimal_text(float(value))
    return result


def score_fixture_case(verified: VerifiedScorer, case: dict[str, Any]) -> dict[str, Any]:
    required = {
        "estimated_number_of_nodes",
        "max_distance",
        "prediction",
        "scale",
        "truth",
    }
    if not isinstance(case, dict) or set(case) != required:
        raise ScorerVerificationError("fixture_schema_invalid", "case keys changed")
    estimate = _finite_number(case["estimated_number_of_nodes"], "estimated_number_of_nodes")
    if estimate <= 0:
        raise ScorerVerificationError("fixture_schema_invalid", "estimated node count must be positive")
    scale_value = case["scale"]
    if not isinstance(scale_value, list) or len(scale_value) != 3:
        raise ScorerVerificationError("fixture_schema_invalid", "scale must be z/y/x")
    scale = tuple(_finite_number(item, "scale") for item in scale_value)
    maximum = _finite_number(case["max_distance"], "max_distance")
    if maximum <= 0:
        raise ScorerVerificationError("fixture_schema_invalid", "max distance must be positive")

    prediction = _graph_from_spec(verified, case["prediction"])
    truth = _graph_from_spec(verified, case["truth"])
    scored_prediction = prediction.copy()
    scored_truth = truth.copy()
    official_result = verified.evaluate(
        scored_prediction,
        scored_truth,
        scale=scale,
        max_distance=maximum,
    )
    recall = verified.node_recall(scored_prediction, scored_truth)
    row = verified.per_sample_metrics(official_result, estimate, recall)
    summary = verified.summarise([row])
    counts = {name: int(getattr(official_result, name)) for name in _COUNT_FIELDS}
    return {
        "official_counts": counts,
        "official_summary": _canonical_summary(summary),
    }


def run_fixture_tracer(
    verified: VerifiedScorer,
    fixture_path: str | Path,
    expected_path: str | Path,
    *,
    case_name: str = "perfect_linear",
) -> dict[str, Any]:
    fixture = _load_json(fixture_path)
    expected = _load_json(expected_path)
    cases = fixture.get("cases")
    expected_cases = expected.get("cases")
    if fixture.get("schema_version") != 1 or not isinstance(cases, dict) or case_name not in cases:
        raise ScorerVerificationError("fixture_schema_invalid", case_name)
    if expected.get("schema_version") != 1 or not isinstance(expected_cases, dict) or case_name not in expected_cases:
        raise ScorerVerificationError("fixture_expected_invalid", case_name)
    expected_case = expected_cases[case_name]
    expected_hash = sha256_bytes(canonical_json_bytes(expected_case))
    if case_name == "perfect_linear" and not secrets.compare_digest(
        expected_hash, verified.lock.fixture_result_sha256
    ):
        raise ScorerVerificationError("fixture_result_hash_mismatch", case_name)

    measured = score_fixture_case(verified, cases[case_name])
    if measured != expected_case:
        raise ScorerVerificationError("fixture_result_mismatch", case_name)
    return {
        "adapter_schema": verified.lock.raw["api"]["adapter_schema"],
        "call_chain": ["evaluate", "per_sample_metrics", "summarise"],
        "fixture_case": case_name,
        "fixture_result_sha256": expected_hash,
        "official_counts": measured["official_counts"],
        "official_summary": measured["official_summary"],
        "private_container_parity": "unproven",
        "scorer_lock_sha256": verified.lock_sha256,
        "source_parity": "verified",
    }
