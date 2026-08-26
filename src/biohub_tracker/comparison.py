from __future__ import annotations

import math
import warnings
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Sequence

from .io import canonical_json_bytes, sha256_bytes


class ComparisonError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(detail: str) -> None:
    raise ComparisonError("COMPARISON_BOUNDARY_MISMATCH", detail)


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ComparisonError("NONFINITE_COMPARISON", name)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ComparisonError("NONFINITE_COMPARISON", name) from exc
    if not result.is_finite():
        raise ComparisonError("NONFINITE_COMPARISON", name)
    return result


def _text(value: Any, name: str) -> str:
    result = _decimal(value, name)
    if result == 0:
        return "0"
    return format(result.normalize(), "f")


def _sha(value: Any, name: str) -> str:
    text = str(value).casefold()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        _fail(name)
    return text


def _delta(baseline: Any, candidate: Any, name: str) -> dict[str, Any]:
    if baseline is None or candidate is None:
        if baseline is None and candidate is None:
            return {
                "status": "not_applicable",
                "reason": "no_events_in_either_role",
                "baseline": None,
                "candidate": None,
                "delta": None,
            }
        return {
            "status": "not_comparable",
            "reason": "event_support_differs_between_roles",
            "baseline": baseline,
            "candidate": candidate,
            "delta": None,
        }
    baseline_value = _decimal(baseline, f"{name}.baseline")
    candidate_value = _decimal(candidate, f"{name}.candidate")
    return {
        "status": "applicable",
        "reason": None,
        "baseline": _text(baseline_value, f"{name}.baseline"),
        "candidate": _text(candidate_value, f"{name}.candidate"),
        "delta": _text(candidate_value - baseline_value, f"{name}.delta"),
    }


def _count_delta(baseline: Any, candidate: Any, name: str) -> dict[str, int]:
    if isinstance(baseline, bool) or isinstance(candidate, bool):
        _fail(name)
    baseline_value, candidate_value = int(baseline), int(candidate)
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": candidate_value - baseline_value,
    }


def _group_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        name: _delta(baseline.get(name), candidate.get(name), name)
        for name in (
            "score",
            "adjusted_edge_jaccard",
            "edge_jaccard",
            "division_jaccard",
            "organizer_macro_node_recall",
            "node_recall_micro",
        )
    }
    baseline_counts = baseline.get("official_counts")
    candidate_counts = candidate.get("official_counts")
    if not isinstance(baseline_counts, Mapping) or not isinstance(candidate_counts, Mapping):
        _fail("official_counts")
    if set(baseline_counts) != set(candidate_counts):
        _fail("official count fields")
    counts = {
        name: _count_delta(baseline_counts[name], candidate_counts[name], name)
        for name in sorted(baseline_counts)
    }
    counts.update(
        {
            name: _count_delta(baseline[name], candidate[name], name)
            for name in ("gt_node_count", "matched_gt_node_count")
        }
    )
    counts["estimated_number_of_nodes"] = _delta(
        baseline["estimated_number_of_nodes"],
        candidate["estimated_number_of_nodes"],
        "estimated_number_of_nodes",
    )
    return {"metrics": metrics, "counts": counts}


def _diagnostic_scalar(value: Mapping[str, Any], name: str) -> Any:
    if value.get("status") not in {"applicable", "not_applicable"}:
        _fail(name)
    return value.get("value")


def _diagnostic_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        baseline.get("authority") != "non_authoritative_diagnostic"
        or candidate.get("authority") != "non_authoritative_diagnostic"
        or baseline.get("organizer_input_eligible") is not False
        or candidate.get("organizer_input_eligible") is not False
    ):
        _fail("diagnostic authority")
    values = {}
    for name in (
        "endpoint_availability",
        "conditional_association_recall",
        "conditional_valid_edge_precision",
        "conditional_valid_edge_jaccard",
        "oracle_gap_adjusted_edge",
    ):
        if not isinstance(baseline.get(name), Mapping) or not isinstance(
            candidate.get(name), Mapping
        ):
            _fail(f"required diagnostic field {name}")
        values[name] = (
            _diagnostic_scalar(baseline[name], name),
            _diagnostic_scalar(candidate[name], name),
        )
    for role, diagnostic in (("baseline", baseline), ("candidate", candidate)):
        oracle = diagnostic.get("oracle_link_ceiling")
        if not isinstance(oracle, Mapping):
            _fail(f"required diagnostic field {role}.oracle_link_ceiling")
        for component in ("raw", "adjusted"):
            if not isinstance(oracle.get(component), Mapping):
                _fail(f"required diagnostic field {role}.oracle_link_ceiling.{component}")
    values["oracle_link_ceiling_raw"] = (
        _diagnostic_scalar(baseline["oracle_link_ceiling"]["raw"], "oracle.raw"),
        _diagnostic_scalar(candidate["oracle_link_ceiling"]["raw"], "oracle.raw"),
    )
    values["oracle_link_ceiling_adjusted"] = (
        _diagnostic_scalar(baseline["oracle_link_ceiling"]["adjusted"], "oracle.adjusted"),
        _diagnostic_scalar(candidate["oracle_link_ceiling"]["adjusted"], "oracle.adjusted"),
    )
    if "node_count_ratio" not in baseline or "node_count_ratio" not in candidate:
        _fail("required diagnostic field node_count_ratio")
    values["node_count_ratio"] = (
        baseline["node_count_ratio"], candidate["node_count_ratio"]
    )
    result = {
        name: _delta(pair[0], pair[1], f"diagnostic.{name}")
        for name, pair in sorted(values.items())
    }
    strata: dict[str, Any] = {}
    specifications = (
        ("displacement", "bin", ("gt_edges", "endpoint_available", "recovered")),
        ("density", "bin", ("movie_count", "gt_edges", "edge_tp", "edge_fp", "edge_fn")),
        ("divisions", "category", ("count",)),
    )
    for section, identity, fields in specifications:
        baseline_section = baseline.get(section)
        candidate_section = candidate.get(section)
        if not isinstance(baseline_section, Mapping) or not isinstance(candidate_section, Mapping):
            continue
        baseline_rows = {
            str(row[identity]): row for row in baseline_section.get("rows", [])
        }
        candidate_rows = {
            str(row[identity]): row for row in candidate_section.get("rows", [])
        }
        rows = []
        for key in sorted(set(baseline_rows) | set(candidate_rows)):
            baseline_row = baseline_rows.get(key, {})
            candidate_row = candidate_rows.get(key, {})
            rows.append(
                {
                    identity: key,
                    "count_deltas": {
                        field: _count_delta(
                            baseline_row.get(field, 0), candidate_row.get(field, 0), field
                        )
                        for field in fields
                    },
                }
            )
        strata[section] = rows
    if strata:
        result["strata"] = strata
    return result


def _pooled_diagnostic(movies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ratio_names = (
        "endpoint_availability",
        "conditional_association_recall",
        "conditional_valid_edge_precision",
        "conditional_valid_edge_jaccard",
    )
    ratios = {name: [0, 0] for name in ratio_names}
    predicted_nodes = Decimal(0)
    estimated_nodes = Decimal(0)
    adjusted_edge_weighted_sum = Decimal(0)
    adjusted_edge_weight = 0
    displacement: dict[str, dict[str, Any]] = {}
    density: dict[str, dict[str, Any]] = {}
    divisions: dict[str, int] = {}
    for row in movies.values():
        diagnostic = row["diagnostic_state"]
        for name in ratio_names:
            value = diagnostic.get(name)
            if not isinstance(value, Mapping):
                _fail(f"required diagnostic field {name}")
            ratios[name][0] += int(value["numerator"])
            ratios[name][1] += int(value["denominator"])
        counts = row.get("official_counts")
        if not isinstance(counts, Mapping):
            _fail("official_counts")
        predicted_nodes += _decimal(counts["num_pred_nodes"], "num_pred_nodes")
        estimated_nodes += _decimal(
            row["estimated_number_of_nodes"], "estimated_number_of_nodes"
        )
        edge_weight = sum(int(counts[name]) for name in ("edge_tp", "edge_fp", "edge_fn"))
        adjusted_edge_weight += edge_weight
        adjusted_edge_weighted_sum += (
            _decimal(row["adjusted_edge_jaccard"], "adjusted_edge_jaccard")
            * Decimal(edge_weight)
        )
        for item in diagnostic.get("displacement", {}).get("rows", []):
            target = displacement.setdefault(
                str(item["bin"]),
                {"bin": str(item["bin"]), "gt_edges": 0, "endpoint_available": 0, "recovered": 0},
            )
            for field in ("gt_edges", "endpoint_available", "recovered"):
                target[field] += int(item[field])
        for item in diagnostic.get("density", {}).get("rows", []):
            target = density.setdefault(
                str(item["bin"]),
                {"bin": str(item["bin"]), "movie_count": 0, "gt_edges": 0, "edge_tp": 0, "edge_fp": 0, "edge_fn": 0},
            )
            for field in ("movie_count", "gt_edges", "edge_tp", "edge_fp", "edge_fn"):
                target[field] += int(item[field])
        for item in diagnostic.get("divisions", {}).get("rows", []):
            category = str(item["category"])
            divisions[category] = divisions.get(category, 0) + int(item["count"])
    def pooled_ratio(name: str, reason: str) -> dict[str, Any]:
        numerator, denominator = ratios[name]
        return {
            "status": "applicable" if denominator else "not_applicable",
            "reason": None if denominator else reason,
            "numerator": numerator,
            "denominator": denominator,
            "value": _text(Decimal(numerator) / Decimal(denominator), f"pooled.{name}")
            if denominator
            else None,
        }

    endpoint = pooled_ratio("endpoint_availability", "no_gt_edges")
    if estimated_nodes <= 0:
        _fail("estimated_number_of_nodes")
    node_ratio = (predicted_nodes - estimated_nodes) / estimated_nodes
    if endpoint["value"] is None:
        oracle_adjusted = {
            "status": "not_applicable",
            "reason": "no_gt_edges",
            "value": None,
        }
        oracle_gap = dict(oracle_adjusted)
    else:
        penalty = max(Decimal(0), Decimal(1) - Decimal("0.1") * node_ratio)
        oracle_adjusted_value = _decimal(endpoint["value"], "pooled.oracle.raw") * penalty
        oracle_adjusted = {
            "status": "applicable",
            "reason": None,
            "value": _text(oracle_adjusted_value, "pooled.oracle.adjusted"),
        }
        if adjusted_edge_weight <= 0:
            _fail("adjusted edge weight")
        official_adjusted = adjusted_edge_weighted_sum / Decimal(adjusted_edge_weight)
        oracle_gap = {
            "status": "applicable",
            "reason": None,
            "value": _text(oracle_adjusted_value - official_adjusted, "pooled.oracle.gap"),
        }
    return {
        "authority": "non_authoritative_diagnostic",
        "organizer_input_eligible": False,
        "endpoint_availability": endpoint,
        "conditional_association_recall": pooled_ratio(
            "conditional_association_recall", "no_available_gt_edges"
        ),
        "conditional_valid_edge_precision": pooled_ratio(
            "conditional_valid_edge_precision", "no_valid_prediction_edges"
        ),
        "conditional_valid_edge_jaccard": pooled_ratio(
            "conditional_valid_edge_jaccard", "empty_conditional_union"
        ),
        "node_count_ratio": _text(node_ratio, "pooled.node_count_ratio"),
        "oracle_link_ceiling": {
            "raw": dict(endpoint),
            "adjusted": oracle_adjusted,
        },
        "oracle_gap_adjusted_edge": oracle_gap,
        "displacement": {"rows": [displacement[key] for key in sorted(displacement)]},
        "density": {"rows": [density[key] for key in sorted(density)]},
        "divisions": {
            "rows": [
                {"category": key, "count": divisions[key]} for key in sorted(divisions)
            ]
        },
    }


def _movie_group_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = {
        name: _delta(baseline.get(name), candidate.get(name), name)
        for name in (
            "score",
            "adjusted_edge_jaccard",
            "edge_jaccard",
            "division_jaccard",
            "node_recall",
            "node_count_ratio",
        )
    }
    baseline_counts = baseline["official_counts"]
    candidate_counts = candidate["official_counts"]
    if set(baseline_counts) != set(candidate_counts):
        _fail("movie official count fields")
    counts = {
        name: _count_delta(baseline_counts[name], candidate_counts[name], name)
        for name in sorted(baseline_counts)
    }
    counts.update(
        {
            name: _count_delta(baseline[name], candidate[name], name)
            for name in ("gt_node_count", "matched_gt_node_count")
        }
    )
    return {"metrics": metrics, "counts": counts}


def _validate_boundary(
    *,
    registration: Mapping[str, Any],
    aggregate_status: str,
    members: Sequence[Mapping[str, Any]],
    authoritative_inventories: Sequence[Mapping[str, Any]],
    baseline_movies: Sequence[Mapping[str, Any]],
    candidate_movies: Sequence[Mapping[str, Any]],
    expected_sample_ids: Sequence[str],
    policy_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if aggregate_status != "running":
        _fail(f"aggregate lifecycle:{aggregate_status}")
    if registration.get("evaluation_policy_sha256") != _sha(policy_sha256, "policy_sha256"):
        _fail("evaluation policy")
    for name in ("scorer_lock_sha256", "environment_lock_sha256", "manifest_sha256"):
        _sha(registration.get(name), name)
    canonical_members = sorted(
        (dict(item) for item in members), key=lambda item: (item["role"], item["fold_id"])
    )
    if canonical_json_bytes(registration.get("members")) != canonical_json_bytes(canonical_members):
        _fail("registered member table")
    if len(canonical_members) != 4:
        _fail("four reciprocal members required")
    folds = sorted({str(item["fold_id"]) for item in canonical_members})
    expected_slots = {(role, fold) for role in ("baseline", "candidate") for fold in folds}
    actual_slots = {(str(item["role"]), str(item["fold_id"])) for item in canonical_members}
    if len(folds) != 2 or actual_slots != expected_slots:
        _fail("member slots")
    member_by_slot = {
        (str(item["role"]), str(item["fold_id"])): item for item in canonical_members
    }
    for member in canonical_members:
        for name in (
            "producer_registration_event_sha256",
            "producer_terminal_event_sha256",
            "manifest_sha256",
            "train_membership_sha256",
            "calibration_membership_sha256",
            "evaluation_membership_sha256",
            "model_sha256",
            "config_sha256",
            "code_sha256",
            "data_sha256",
            "graph_inventory_sha256",
        ):
            _sha(member.get(name), f"member.{name}")
        artifacts = member.get("artifact_hashes")
        if not isinstance(artifacts, Mapping) or not artifacts:
            _fail("member artifact hashes")
        for name, digest in sorted(artifacts.items()):
            _sha(digest, f"member.artifact_hashes.{name}")

    inventories = sorted(
        (dict(item) for item in authoritative_inventories),
        key=lambda item: (item["role"], item["fold_id"]),
    )
    if len(inventories) != 4 or {
        (str(item["role"]), str(item["fold_id"])) for item in inventories
    } != expected_slots:
        _fail("authoritative inventory slots")
    for item in inventories:
        slot = (str(item["role"]), str(item["fold_id"]))
        member = member_by_slot[slot]
        if item.get("producer_run_id") != member.get("producer_run_id"):
            _fail("authoritative inventory producer")
        for name in (
            "submission_graph_inventory_sha256",
            "roundtrip_evidence_sha256",
            "csv_sha256",
        ):
            _sha(item.get(name), f"inventory.{name}")

    expected = sorted(str(item) for item in expected_sample_ids)
    if len(expected) != len(set(expected)):
        _fail("duplicate expected samples")

    def role_map(role: str, movies: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for item in movies:
            sample_id = str(item.get("sample_id"))
            if sample_id in result:
                _fail(f"duplicate {role} sample:{sample_id}")
            fold_id = str(item.get("fold_id"))
            producer = item.get("producer")
            if not isinstance(producer, Mapping) or canonical_json_bytes(producer) != canonical_json_bytes(
                member_by_slot.get((role, fold_id))
            ):
                _fail(f"{role} producer binding:{sample_id}")
            if "organizer_row" not in item or "diagnostic_state" not in item:
                _fail(f"{role} evidence row:{sample_id}")
            result[sample_id] = item
        if sorted(result) != expected:
            _fail(f"{role} coverage")
        return result

    baseline = role_map("baseline", baseline_movies)
    candidate = role_map("candidate", candidate_movies)
    for sample_id in expected:
        if (
            baseline[sample_id].get("embryo_id") != candidate[sample_id].get("embryo_id")
            or baseline[sample_id].get("fold_id") != candidate[sample_id].get("fold_id")
        ):
            _fail(f"paired identity:{sample_id}")
    return baseline, candidate


def paired_movie_bootstrap(
    *,
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    aggregate: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
) -> dict[str, Any]:
    bootstrap = policy.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
        "algorithm",
        "percentiles",
        "percentile_method",
        "repetitions",
        "seed",
        "unit",
    }:
        raise ComparisonError("BOOTSTRAP_POLICY_INVALID", "schema")
    if (
        bootstrap["algorithm"] != "numpy.Generator(PCG64)"
        or bootstrap["unit"] != "complete_movie_within_embryo"
        or bootstrap["percentile_method"] != "linear"
    ):
        raise ComparisonError("BOOTSTRAP_POLICY_INVALID", "semantics")
    repetitions = int(bootstrap["repetitions"])
    seed = int(bootstrap["seed"])
    if repetitions <= 0 or seed < 0:
        raise ComparisonError("BOOTSTRAP_POLICY_INVALID", "count/seed")
    percentile_labels = tuple(str(item) for item in bootstrap["percentiles"])
    percentiles = tuple(float(_decimal(item, "bootstrap.percentile")) for item in percentile_labels)
    if tuple(sorted(percentiles)) != percentiles or any(item < 0 or item > 100 for item in percentiles):
        raise ComparisonError("BOOTSTRAP_POLICY_INVALID", "percentiles")

    numpy = __import__("numpy")
    rng = numpy.random.Generator(numpy.random.PCG64(seed))
    by_embryo: dict[str, list[str]] = {}
    for sample_id, row in sorted(baseline.items()):
        by_embryo.setdefault(str(row["embryo_id"]), []).append(sample_id)
    metric_fields = (
        "score",
        "adjusted_edge_jaccard",
        "edge_jaccard",
        "division_jaccard",
        "organizer_macro_node_recall",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_fields}
    no_division_baseline = no_division_candidate = no_division_both = 0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No divisions present across any sample in this split*"
        )
        for _replicate in range(repetitions):
            sampled: list[str] = []
            for embryo_id in sorted(by_embryo):
                movie_ids = sorted(by_embryo[embryo_id])
                selected = rng.choice(movie_ids, size=len(movie_ids), replace=True)
                sampled.extend(str(item) for item in selected.tolist())
            baseline_group = aggregate([baseline[item] for item in sampled])
            candidate_group = aggregate([candidate[item] for item in sampled])
            baseline_no_division = baseline_group["division_jaccard"] is None
            candidate_no_division = candidate_group["division_jaccard"] is None
            no_division_baseline += int(baseline_no_division)
            no_division_candidate += int(candidate_no_division)
            no_division_both += int(baseline_no_division and candidate_no_division)
            for name in metric_fields:
                baseline_value = baseline_group[name]
                candidate_value = candidate_group[name]
                if baseline_value is None or candidate_value is None:
                    continue
                delta = float(_decimal(candidate_value, f"bootstrap.{name}.candidate")) - float(
                    _decimal(baseline_value, f"bootstrap.{name}.baseline")
                )
                if not math.isfinite(delta):
                    raise ComparisonError("NONFINITE_COMPARISON", f"bootstrap.{name}")
                values[name].append(delta)

    metric_output: dict[str, Any] = {}
    for name in metric_fields:
        samples = values[name]
        if not samples:
            metric_output[name] = {
                "status": "not_applicable",
                "valid_replicates": 0,
                "percentiles": {label: None for label in percentile_labels},
                "probability_candidate_gt_baseline": None,
            }
            continue
        percentile_values = numpy.percentile(
            numpy.asarray(samples, dtype=numpy.float64),
            percentiles,
            method=str(bootstrap["percentile_method"]),
        )
        positive = sum(item > 0 for item in samples)
        metric_output[name] = {
            "status": "applicable",
            "valid_replicates": len(samples),
            "percentiles": {
                label: _text(value, f"bootstrap.{name}.{label}")
                for label, value in zip(percentile_labels, percentile_values.tolist(), strict=True)
            },
            "probability_candidate_gt_baseline": _text(
                Decimal(positive) / Decimal(len(samples)), f"bootstrap.{name}.probability"
            ),
        }
    return {
        "algorithm": bootstrap["algorithm"],
        "seed": seed,
        "repetitions": repetitions,
        "unit": bootstrap["unit"],
        "stratification": "embryo_id",
        "percentile_method": bootstrap["percentile_method"],
        "metrics": metric_output,
        "no_division_replicate_count": no_division_both,
        "no_division_replicates_by_role": {
            "baseline": no_division_baseline,
            "candidate": no_division_candidate,
        },
        "interpretation": "paired_movie_stability_not_population_proof",
    }


def build_paired_comparison(
    *,
    registration: Mapping[str, Any],
    aggregate_status: str,
    members: Sequence[Mapping[str, Any]],
    authoritative_inventories: Sequence[Mapping[str, Any]],
    baseline_movies: Sequence[Mapping[str, Any]],
    candidate_movies: Sequence[Mapping[str, Any]],
    expected_sample_ids: Sequence[str],
    policy: Mapping[str, Any],
    policy_sha256: str,
    aggregate: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
) -> dict[str, Any]:
    baseline, candidate = _validate_boundary(
        registration=registration,
        aggregate_status=aggregate_status,
        members=members,
        authoritative_inventories=authoritative_inventories,
        baseline_movies=baseline_movies,
        candidate_movies=candidate_movies,
        expected_sample_ids=expected_sample_ids,
        policy_sha256=policy_sha256,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No divisions present across any sample in this split*"
        )
        pooled_baseline = aggregate([baseline[key] for key in sorted(baseline)])
        pooled_candidate = aggregate([candidate[key] for key in sorted(candidate)])

    paired_movies: list[dict[str, Any]] = []
    for sample_id in sorted(baseline):
        baseline_row, candidate_row = baseline[sample_id], candidate[sample_id]
        paired_movies.append(
            {
                "sample_id": sample_id,
                "embryo_id": baseline_row["embryo_id"],
                "fold_id": baseline_row["fold_id"],
                **_movie_group_delta(baseline_row, candidate_row),
                "diagnostic_deltas": _diagnostic_delta(
                    baseline_row["diagnostic_state"], candidate_row["diagnostic_state"]
                ),
            }
        )

    def groups(group_name: str) -> dict[str, Any]:
        keys = sorted({str(row[group_name]) for row in baseline.values()})
        result: dict[str, Any] = {}
        for key in keys:
            sample_ids = sorted(
                sample_id
                for sample_id, row in baseline.items()
                if str(row[group_name]) == key
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="No divisions present across any sample in this split*"
                )
                result[key] = _group_delta(
                    aggregate([baseline[item] for item in sample_ids]),
                    aggregate([candidate[item] for item in sample_ids]),
                )
            result[key]["diagnostic_deltas"] = _diagnostic_delta(
                _pooled_diagnostic({item: baseline[item] for item in sample_ids}),
                _pooled_diagnostic({item: candidate[item] for item in sample_ids}),
            )
        return result

    worst = min(
        paired_movies,
        key=lambda row: (
            _decimal(row["metrics"]["score"]["delta"], "worst_movie_delta"),
            row["sample_id"],
        ),
    )
    lowest_sample_id = min(
        candidate,
        key=lambda sample_id: (
            _decimal(candidate[sample_id]["score"], "candidate.score"), sample_id
        ),
    )
    bootstrap = paired_movie_bootstrap(
        baseline=baseline,
        candidate=candidate,
        policy=policy,
        aggregate=aggregate,
    )
    return {
        "schema_version": "biohub.exact-comparison.v1",
        "direction": "candidate_minus_baseline",
        "member_binding_sha256": sha256_bytes(
            canonical_json_bytes(
                sorted(
                    (dict(item) for item in members),
                    key=lambda item: (item["role"], item["fold_id"]),
                )
            )
        ),
        "authoritative_inventory_binding_sha256": sha256_bytes(
            canonical_json_bytes(
                sorted(
                    (dict(item) for item in authoritative_inventories),
                    key=lambda item: (item["role"], item["fold_id"]),
                )
            )
        ),
        "pooled": {
            **_group_delta(pooled_baseline, pooled_candidate),
            "diagnostic_deltas": _diagnostic_delta(
                _pooled_diagnostic(baseline),
                _pooled_diagnostic(candidate),
            ),
        },
        "by_embryo": groups("embryo_id"),
        "by_fold": groups("fold_id"),
        "by_movie": paired_movies,
        "worst_paired_movie": worst,
        "lowest_absolute_candidate_movie": {
            "sample_id": lowest_sample_id,
            "embryo_id": candidate[lowest_sample_id]["embryo_id"],
            "fold_id": candidate[lowest_sample_id]["fold_id"],
            "score": candidate[lowest_sample_id]["score"],
            "components": {
                name: candidate[lowest_sample_id].get(name)
                for name in (
                    "adjusted_edge_jaccard",
                    "edge_jaccard",
                    "division_jaccard",
                    "node_recall",
                    "node_count_ratio",
                )
            },
            "official_counts": dict(candidate[lowest_sample_id]["official_counts"]),
        },
        "bootstrap": bootstrap,
        "integrity_checks": {
            "aggregate_lifecycle": "running_then_completed_on_report_attachment",
            "member_slots": "passed",
            "producer_event_and_lineage_hashes": "passed",
            "authoritative_roundtrip_binding": "passed",
            "paired_complete_movie_coverage": "passed",
            "external_presentation_metadata_excluded": "passed",
        },
    }
