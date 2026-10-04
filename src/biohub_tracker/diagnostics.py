from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from .graphs import GraphData, GraphNode


class DiagnosticError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(detail: str) -> None:
    raise DiagnosticError("DIAGNOSTIC_RECONCILIATION_FAILED", detail)


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise DiagnosticError("NONFINITE_DIAGNOSTIC", name)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DiagnosticError("NONFINITE_DIAGNOSTIC", name) from exc
    if not result.is_finite():
        raise DiagnosticError("NONFINITE_DIAGNOSTIC", name)
    return result


def _text(value: Any, name: str) -> str:
    result = _decimal(value, name)
    if result == 0:
        return "0"
    return format(result.normalize(), "f")


def _ratio(numerator: int, denominator: int, *, reason: str) -> dict[str, Any]:
    if denominator == 0:
        return {
            "status": "not_applicable",
            "reason": reason,
            "numerator": numerator,
            "denominator": denominator,
            "value": None,
        }
    return {
        "status": "applicable",
        "reason": None,
        "numerator": numerator,
        "denominator": denominator,
        "value": _text(Decimal(numerator) / Decimal(denominator), reason),
    }


def _normalized_nodes(graph: GraphData, name: str) -> dict[int, GraphNode]:
    result: dict[int, GraphNode] = {}
    for node in graph.nodes:
        try:
            node_id = int(node.node_id)
        except (TypeError, ValueError) as exc:
            raise DiagnosticError("DIAGNOSTIC_GRAPH_INVALID", f"{name}:node_id") from exc
        if node_id in result:
            raise DiagnosticError("DIAGNOSTIC_GRAPH_INVALID", f"{name}:duplicate:{node_id}")
        result[node_id] = node
    return result


def _normalized_edges(graph: GraphData, nodes: Mapping[int, GraphNode], name: str) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for raw_source, raw_target in graph.edges:
        try:
            edge = (int(raw_source), int(raw_target))
        except (TypeError, ValueError) as exc:
            raise DiagnosticError("DIAGNOSTIC_GRAPH_INVALID", f"{name}:edge") from exc
        if edge[0] not in nodes or edge[1] not in nodes or edge in result:
            raise DiagnosticError("DIAGNOSTIC_GRAPH_INVALID", f"{name}:edge:{edge}")
        result.add(edge)
    return result


def _boundaries(values: Sequence[Any], name: str) -> tuple[Decimal, ...]:
    result = tuple(_decimal(item, name) for item in values)
    if not result or any(item <= 0 for item in result) or tuple(sorted(set(result))) != result:
        raise DiagnosticError("DIAGNOSTIC_POLICY_INVALID", name)
    return result


def _bin_label(value: Decimal, boundaries: tuple[Decimal, ...], suffix: str) -> str:
    for index, boundary in enumerate(boundaries):
        if value < boundary:
            lower = "0" if index == 0 else _text(boundaries[index - 1], "boundary")
            return f"{lower}_to_lt_{_text(boundary, 'boundary')}_{suffix}"
    return f"ge_{_text(boundaries[-1], 'boundary')}_{suffix}"


def _sum_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, int]:
    return {field: sum(int(row[field]) for row in rows) for field in fields}


def _descendants(adjacency: Mapping[int, Sequence[int]], start: int) -> set[int]:
    result: set[int] = set()
    pending = [start]
    while pending:
        node = pending.pop()
        if node in result:
            continue
        result.add(node)
        pending.extend(adjacency.get(node, ()))
    return result


def _pair_division_forks(
    *,
    pred_edges: set[tuple[int, int]],
    truth_edges: set[tuple[int, int]],
    matches: Mapping[int, int],
    tp_forks: Sequence[int],
    recovered_divisions: Sequence[int],
) -> list[tuple[int, int]]:
    pred_children: dict[int, list[int]] = defaultdict(list)
    truth_children: dict[int, list[int]] = defaultdict(list)
    for source, target in pred_edges:
        pred_children[source].append(target)
    for source, target in truth_edges:
        truth_children[source].append(target)
    truth_branches = {
        division: [_descendants(truth_children, child) for child in truth_children[division]]
        for division in recovered_divisions
    }
    pairs: list[tuple[int, int]] = []
    used_truth: set[int] = set()
    for fork in sorted(tp_forks):
        children = pred_children.get(fork, [])
        if len(children) != 2:
            _fail("division TP fork must have exactly two children")
        mapped_branches = [
            {
                matches[node]
                for node in _descendants(pred_children, child)
                if node in matches
            }
            for child in children
        ]
        candidates: list[int] = []
        for division, branches in truth_branches.items():
            direct = matches.get(fork) == division
            branch_match = (
                bool(mapped_branches[0] & branches[0])
                and bool(mapped_branches[1] & branches[1])
            ) or (
                bool(mapped_branches[0] & branches[1])
                and bool(mapped_branches[1] & branches[0])
            )
            if direct or branch_match:
                candidates.append(division)
        if len(candidates) != 1 or candidates[0] in used_truth:
            _fail("division correspondence missing, ambiguous, or duplicated")
        used_truth.add(candidates[0])
        pairs.append((candidates[0], fork))
    if used_truth != set(recovered_divisions):
        _fail("division correspondence does not cover recovered truth forks")
    return pairs


def diagnose_movie(
    *,
    prediction: GraphData,
    truth: GraphData,
    prediction_to_truth: Mapping[int, int],
    official_counts: Mapping[str, int],
    official_adjusted_edge_jaccard: Any,
    estimated_number_of_nodes: Any,
    shape_tzyx: Sequence[int],
    scale_zyx_um: Sequence[Any],
    adjustment_alpha: Any,
    displacement_boundaries_um: Sequence[Any],
    density_boundaries_per_mm3: Sequence[Any],
    density_boundary_rule: str = "training-side-frozen-per-manifest-v1",
    division_offsets_frames: Sequence[int] = (-1, 0, 1),
    no_event_encoding: str = "not_applicable",
    division_scores: Mapping[int, int] | None = None,
    division_tp_forks: Sequence[int] = (),
    division_fp_forks: Sequence[int] = (),
) -> dict[str, Any]:
    """Explain one official scoring row without supplying organizer score inputs."""

    if density_boundary_rule != "training-side-frozen-per-manifest-v1":
        raise DiagnosticError("DIAGNOSTIC_POLICY_INVALID", "density_boundary_rule")
    if tuple(int(item) for item in division_offsets_frames) != (-1, 0, 1):
        raise DiagnosticError("DIAGNOSTIC_POLICY_INVALID", "division_offsets_frames")
    if no_event_encoding != "not_applicable":
        raise DiagnosticError("DIAGNOSTIC_POLICY_INVALID", "no_event_encoding")
    pred_nodes = _normalized_nodes(prediction, "prediction")
    truth_nodes = _normalized_nodes(truth, "truth")
    pred_edges = _normalized_edges(prediction, pred_nodes, "prediction")
    truth_edges = _normalized_edges(truth, truth_nodes, "truth")
    matches = {
        int(pred_id): int(truth_id)
        for pred_id, truth_id in prediction_to_truth.items()
        if int(truth_id) >= 0
    }
    if any(pred_id not in pred_nodes or truth_id not in truth_nodes for pred_id, truth_id in matches.items()):
        raise DiagnosticError("DIAGNOSTIC_MATCH_INVALID", "match endpoint missing")

    required_counts = {
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
        "num_pred_nodes",
    }
    if set(official_counts) != required_counts:
        _fail("official count schema")
    counts = {name: int(official_counts[name]) for name in sorted(required_counts)}
    if any(value < 0 for value in counts.values()) or counts["num_pred_nodes"] != len(pred_nodes):
        _fail("official counts")
    if counts["edge_tp"] + counts["edge_fn"] != len(truth_edges):
        _fail("GT edge total")

    matched_truth = set(matches.values())
    available_edges = {
        edge for edge in truth_edges if edge[0] in matched_truth and edge[1] in matched_truth
    }
    recovered_edges = {
        (matches[source], matches[target])
        for source, target in pred_edges
        if source in matches and target in matches and (matches[source], matches[target]) in truth_edges
    }
    if len(recovered_edges) != counts["edge_tp"]:
        _fail(f"recovered edge count expected {counts['edge_tp']}, got {len(recovered_edges)}")
    if not recovered_edges <= available_edges:
        _fail("recovered edges unavailable")

    estimate = _decimal(estimated_number_of_nodes, "estimated_number_of_nodes")
    if estimate <= 0:
        raise DiagnosticError("DIAGNOSTIC_POLICY_INVALID", "estimated_number_of_nodes")
    alpha = _decimal(adjustment_alpha, "adjustment_alpha")
    node_ratio = (Decimal(len(pred_nodes)) - estimate) / estimate
    raw_oracle = _ratio(len(available_edges), len(truth_edges), reason="no_gt_edges")
    if raw_oracle["value"] is None:
        adjusted_oracle: dict[str, Any] = {
            "status": "not_applicable",
            "reason": "no_gt_edges",
            "value": None,
        }
        oracle_gap: dict[str, Any] = dict(adjusted_oracle)
    else:
        penalty = max(Decimal(0), Decimal(1) - alpha * node_ratio)
        adjusted = _decimal(raw_oracle["value"], "oracle_raw") * penalty
        adjusted_oracle = {"status": "applicable", "reason": None, "value": _text(adjusted, "oracle_adjusted")}
        gap = adjusted - _decimal(official_adjusted_edge_jaccard, "official_adjusted_edge_jaccard")
        oracle_gap = {"status": "applicable", "reason": None, "value": _text(gap, "oracle_gap")}

    displacement_boundaries = _boundaries(displacement_boundaries_um, "displacement_boundaries_um")
    displacement: dict[str, dict[str, Any]] = {}
    scale = tuple(_decimal(item, "scale_zyx_um") for item in scale_zyx_um)
    if len(scale) != 3 or any(item <= 0 for item in scale):
        raise DiagnosticError("DIAGNOSTIC_POLICY_INVALID", "scale_zyx_um")
    for edge in sorted(truth_edges):
        source, target = (truth_nodes[edge[0]], truth_nodes[edge[1]])
        squared = sum(
            ((_decimal(getattr(target, axis), axis) - _decimal(getattr(source, axis), axis)) * scale[index]) ** 2
            for index, axis in enumerate(("z", "y", "x"))
        )
        distance = squared.sqrt()
        label = _bin_label(distance, displacement_boundaries, "um")
        row = displacement.setdefault(
            label,
            {"bin": label, "gt_edges": 0, "endpoint_available": 0, "recovered": 0},
        )
        row["gt_edges"] += 1
        row["endpoint_available"] += int(edge in available_edges)
        row["recovered"] += int(edge in recovered_edges)
    displacement_rows = [displacement[key] for key in sorted(displacement)]
    displacement_totals = _sum_rows(
        displacement_rows, ("gt_edges", "endpoint_available", "recovered")
    )
    expected_displacement = {
        "gt_edges": len(truth_edges),
        "endpoint_available": len(available_edges),
        "recovered": counts["edge_tp"],
    }
    if displacement_totals != expected_displacement:
        _fail("displacement coverage")

    if len(shape_tzyx) != 4 or any(int(item) <= 0 for item in shape_tzyx):
        raise DiagnosticError("DIAGNOSTIC_POLICY_INVALID", "shape_tzyx")
    timepoints, z_size, y_size, x_size = (int(item) for item in shape_tzyx)
    physical_movie_volume = (
        Decimal(timepoints)
        * Decimal(z_size)
        * scale[0]
        * Decimal(y_size)
        * scale[1]
        * Decimal(x_size)
        * scale[2]
    )
    density = estimate * Decimal(1_000_000_000) / physical_movie_volume
    density_boundaries = _boundaries(density_boundaries_per_mm3, "density_boundaries_per_mm3")
    density_label = _bin_label(density, density_boundaries, "per_mm3")
    density_rows = [
        {
            "bin": density_label,
            "density_per_mm3": _text(density, "density_per_mm3"),
            "movie_count": 1,
            "gt_edges": len(truth_edges),
            "edge_tp": counts["edge_tp"],
            "edge_fp": counts["edge_fp"],
            "edge_fn": counts["edge_fn"],
        }
    ]

    truth_outdegree: dict[int, int] = defaultdict(int)
    for source, _target in truth_edges:
        truth_outdegree[source] += 1
    gt_divisions = sorted(node_id for node_id, degree in truth_outdegree.items() if degree == 2)
    normalized_scores = {int(key): int(value) for key, value in (division_scores or {}).items()}
    if set(normalized_scores) != set(gt_divisions) or any(value not in {0, 1} for value in normalized_scores.values()):
        _fail("division score coverage")
    if sum(normalized_scores.values()) != counts["division_tp"]:
        _fail("division TP")
    if len(gt_divisions) - sum(normalized_scores.values()) != counts["division_fn"]:
        _fail("division FN")
    tp_forks = sorted(int(item) for item in set(division_tp_forks))
    fp_forks = sorted(int(item) for item in set(division_fp_forks))
    if len(tp_forks) != counts["division_tp"] or len(fp_forks) != counts["division_fp"]:
        _fail("division fork totals")
    if any(item not in pred_nodes for item in (*tp_forks, *fp_forks)):
        _fail("division fork node")

    recovered_divisions = sorted(key for key, value in normalized_scores.items() if value == 1)
    division_pairs = _pair_division_forks(
        pred_edges=pred_edges,
        truth_edges=truth_edges,
        matches=matches,
        tp_forks=tp_forks,
        recovered_divisions=recovered_divisions,
    )
    timing_counts = {"-1": 0, "0": 0, "+1": 0, "outside_window": 0}
    for gt, pred in division_pairs:
        offset = int(pred_nodes[pred].t) - int(truth_nodes[gt].t)
        key = f"{offset:+d}" if offset in {-1, 1} else str(offset)
        timing_counts[key if key in timing_counts else "outside_window"] += 1
    if len(division_pairs) != counts["division_tp"]:
        _fail("division pairing")
    division_rows = [
        {"category": f"support_offset_{key}", "count": timing_counts[key]}
        for key in ("-1", "0", "+1", "outside_window")
    ]
    division_rows.extend(
        (
            {"category": "missed_gt_division", "count": counts["division_fn"]},
            {"category": "false_positive_predicted_fork", "count": counts["division_fp"]},
        )
    )

    return {
        "authority": "non_authoritative_diagnostic",
        "organizer_input_eligible": False,
        "endpoint_availability": _ratio(len(available_edges), len(truth_edges), reason="no_gt_edges"),
        "oracle_link_ceiling": {"raw": raw_oracle, "adjusted": adjusted_oracle},
        "conditional_association_recall": _ratio(counts["edge_tp"], len(available_edges), reason="no_available_gt_edges"),
        "conditional_valid_edge_precision": _ratio(
            counts["edge_tp"], counts["edge_tp"] + counts["edge_fp"], reason="no_valid_prediction_edges"
        ),
        "conditional_valid_edge_jaccard": _ratio(
            counts["edge_tp"], len(available_edges) + counts["edge_fp"], reason="empty_conditional_union"
        ),
        "node_count_ratio": _text(node_ratio, "node_count_ratio"),
        "oracle_gap_adjusted_edge": oracle_gap,
        "displacement": {
            "boundary_rule": "lower_inclusive_upper_exclusive",
            "rows": displacement_rows,
            "excluded_count": 0,
            "totals": displacement_totals,
        },
        "density": {
            "boundary_rule": density_boundary_rule,
            "rows": density_rows,
            "excluded_count": 0,
        },
        "divisions": {
            "rows": division_rows,
            "excluded_count": timing_counts["outside_window"],
            "official_counts": {
                "division_tp": counts["division_tp"],
                "division_fp": counts["division_fp"],
                "division_fn": counts["division_fn"],
            },
            "no_event": counts["division_tp"] + counts["division_fp"] + counts["division_fn"] == 0,
        },
        "reconciliation": {
            "status": "passed",
            "edge_tp": counts["edge_tp"],
            "edge_fp": counts["edge_fp"],
            "edge_fn": counts["edge_fn"],
            "division_tp": counts["division_tp"],
            "division_fp": counts["division_fp"],
            "division_fn": counts["division_fn"],
        },
    }


def aggregate_movie_diagnostics(movies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not movies:
        raise DiagnosticError("DIAGNOSTIC_RECONCILIATION_FAILED", "empty diagnostic group")
    displacement: dict[str, dict[str, Any]] = {}
    density: dict[str, dict[str, Any]] = {}
    division: dict[str, int] = defaultdict(int)
    reconciliation = {name: 0 for name in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")}
    endpoint_numerator = endpoint_denominator = 0
    for movie in movies:
        if movie.get("authority") != "non_authoritative_diagnostic" or movie.get("organizer_input_eligible") is not False:
            _fail("diagnostic authority marker")
        endpoint = movie["endpoint_availability"]
        endpoint_numerator += int(endpoint["numerator"])
        endpoint_denominator += int(endpoint["denominator"])
        for row in movie["displacement"]["rows"]:
            target = displacement.setdefault(
                str(row["bin"]),
                {"bin": str(row["bin"]), "gt_edges": 0, "endpoint_available": 0, "recovered": 0},
            )
            for field in ("gt_edges", "endpoint_available", "recovered"):
                target[field] += int(row[field])
        for row in movie["density"]["rows"]:
            target = density.setdefault(
                str(row["bin"]),
                {"bin": str(row["bin"]), "movie_count": 0, "gt_edges": 0, "edge_tp": 0, "edge_fp": 0, "edge_fn": 0},
            )
            for field in ("movie_count", "gt_edges", "edge_tp", "edge_fp", "edge_fn"):
                target[field] += int(row[field])
        for row in movie["divisions"]["rows"]:
            division[str(row["category"])] += int(row["count"])
        for field in reconciliation:
            reconciliation[field] += int(movie["reconciliation"][field])
    if sum(row["recovered"] for row in displacement.values()) != reconciliation["edge_tp"]:
        _fail("aggregate displacement TP")
    if sum(row["edge_fp"] for row in density.values()) != reconciliation["edge_fp"]:
        _fail("aggregate density FP")
    return {
        "authority": "non_authoritative_diagnostic",
        "organizer_input_eligible": False,
        "movie_count": len(movies),
        "endpoint_availability": _ratio(endpoint_numerator, endpoint_denominator, reason="no_gt_edges"),
        "displacement_rows": [displacement[key] for key in sorted(displacement)],
        "density_rows": [density[key] for key in sorted(density)],
        "division_rows": [
            {"category": key, "count": division[key]} for key in sorted(division)
        ],
        "reconciliation": {"status": "passed", **reconciliation},
    }
