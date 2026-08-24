from __future__ import annotations

import importlib
import json
import math
import os
import secrets
import shutil
import tempfile
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from .comparison import build_paired_comparison
from .diagnostics import aggregate_movie_diagnostics, diagnose_movie
from .evidence import resolve_producer
from .graphs import (
    PredictionInventory,
    artifact_tree_sha256,
    graph_data_from_tracksdata,
    load_geff_graph,
    preflight_prediction_set,
)
from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file, utc_now
from .ledger import (
    EventType,
    ExactEvaluationStatus,
    ExperimentEvent,
    Ledger,
    exact_evaluation_completed_payload,
    exact_evaluation_failed_payload,
    exact_evaluation_registration_payload,
    reconstruct_exact_evaluations,
)
from .manifests import EvaluationManifest, SampleRecord, load_manifest
from .scorer_lock import ScorerLock, verify_scorer_lock
from .submission_io import RoundTripEvidence, roundtrip_prediction_inventory


REPORT_SCHEMA = "biohub.exact-report.v1"
_SHA256 = frozenset("0123456789abcdef")
_POLICY_KEYS = {
    "schema_version",
    "report_schema",
    "authoritative_prediction_space",
    "native_prediction_space",
    "diagnostics",
    "bootstrap",
}
_DIAGNOSTIC_POLICY_KEYS = {
    "density_boundaries_per_mm3",
    "density_boundary_rule",
    "displacement_boundaries_um",
    "division_offsets_frames",
    "no_event_encoding",
}
_BOOTSTRAP_POLICY_KEYS = {
    "algorithm",
    "percentiles",
    "percentile_method",
    "repetitions",
    "seed",
    "unit",
}
_CORE_KEYS = {
    "schema_version",
    "evaluation_run_id",
    "evidence_kind",
    "scorer_lock_sha256",
    "environment_lock_sha256",
    "manifest_sha256",
    "evaluation_policy_sha256",
    "members",
    "authoritative_inventories",
    "coverage",
    "official",
    "diagnostics",
    "comparison",
    "integrity_checks",
    "promotion_eligible",
}
_ENVELOPE_KEYS = {
    "schema_version",
    "report_core_sha256",
    "created_at",
    "runtime_seconds",
    "peak_memory_bytes",
    "presentation_metadata",
}


class ExactEvaluationError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> None:
    raise ExactEvaluationError(reason_code, detail)


def _sha(value: Any, name: str) -> str:
    digest = str(value).casefold()
    if len(digest) != 64 or any(char not in _SHA256 for char in digest):
        _fail("EXACT_REPORT_SCHEMA_INVALID", f"{name} must be a SHA-256 digest")
    return digest


def decimal_string(value: Any, name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        _fail("NONFINITE_METRIC", name)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ExactEvaluationError("NONFINITE_METRIC", name) from exc
    if not result.is_finite():
        _fail("NONFINITE_METRIC", name)
    if result == 0:
        return "0"
    return format(result.normalize(), "f")


def _finite_tree(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _finite_tree(value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("NONFINITE_METRIC", path)
    elif isinstance(value, Decimal) and not value.is_finite():
        _fail("NONFINITE_METRIC", path)


def _forbidden_presentation_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in {
                "leaderboard_score",
                "public_leaderboard_score",
                "public_score",
            } or _forbidden_presentation_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_forbidden_presentation_key(item) for item in value)
    return False


@dataclass(frozen=True)
class PredictionSetRef:
    fold_id: str
    graph_dir: Path
    producer_manifest_path: Path


@dataclass(frozen=True)
class ExactEvaluationRequest:
    ledger_path: Path
    evaluation_run_id: str
    truth_dir: Path
    manifest_path: Path
    scorer_lock_path: Path
    evaluation_policy_path: Path
    baseline_sets: tuple[PredictionSetRef, ...]
    candidate_sets: tuple[PredictionSetRef, ...]
    output_dir: Path
    scorer_checkout: Path = Path(".biohub/vendor/kaggle-cell-tracking-competition")
    tracksdata_checkout: Path = Path(".biohub/vendor/tracksdata")
    workspace_root: Path = Path(".")
    presentation_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExactReport:
    core: Mapping[str, Any]
    envelope: Mapping[str, Any]
    core_sha256: str
    envelope_sha256: str
    core_path: Path | None = None
    envelope_path: Path | None = None

    @classmethod
    def from_files(cls, core_path: str | Path, envelope_path: str | Path) -> "ExactReport":
        try:
            core = json.loads(Path(core_path).read_text(encoding="utf-8"))
            envelope = json.loads(Path(envelope_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExactEvaluationError("EXACT_REPORT_UNREADABLE", str(exc)) from exc
        return cls(
            core=core,
            envelope=envelope,
            core_sha256=sha256_bytes(canonical_json_bytes(core)),
            envelope_sha256=sha256_bytes(canonical_json_bytes(envelope)),
            core_path=Path(core_path),
            envelope_path=Path(envelope_path),
        )


def load_evaluation_policy(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactEvaluationError("EVALUATION_POLICY_UNREADABLE", str(path)) from exc
    if not isinstance(value, dict) or set(value) != _POLICY_KEYS:
        _fail("EVALUATION_POLICY_INVALID", "unknown or missing root field")
    if value["schema_version"] != 1 or value["report_schema"] != REPORT_SCHEMA:
        _fail("EVALUATION_POLICY_INVALID", "schema changed")
    if value["authoritative_prediction_space"] != "integer-csv-rebuilt-geff":
        _fail("EVALUATION_POLICY_INVALID", "submission-space authority changed")
    if value["native_prediction_space"] != "diagnostic_only":
        _fail("EVALUATION_POLICY_INVALID", "native-space authority changed")
    diagnostics = value["diagnostics"]
    bootstrap = value["bootstrap"]
    if not isinstance(diagnostics, dict) or not isinstance(bootstrap, dict):
        _fail("EVALUATION_POLICY_INVALID", "policy sections must be objects")
    if set(diagnostics) != _DIAGNOSTIC_POLICY_KEYS:
        _fail("EVALUATION_POLICY_INVALID", "diagnostic policy schema changed")
    if diagnostics["density_boundary_rule"] != "training-side-frozen-per-manifest-v1":
        _fail("EVALUATION_POLICY_INVALID", "density rule changed")
    if diagnostics["division_offsets_frames"] != [-1, 0, 1]:
        _fail("EVALUATION_POLICY_INVALID", "division offsets changed")
    if diagnostics["no_event_encoding"] != "not_applicable":
        _fail("EVALUATION_POLICY_INVALID", "no-event encoding changed")
    if set(bootstrap) != _BOOTSTRAP_POLICY_KEYS:
        _fail("EVALUATION_POLICY_INVALID", "bootstrap policy schema changed")
    if (
        bootstrap["algorithm"] != "numpy.Generator(PCG64)"
        or bootstrap["percentile_method"] != "linear"
        or bootstrap["unit"] != "complete_movie_within_embryo"
        or bootstrap["seed"] != 20260824
        or bootstrap["repetitions"] != 10000
        or bootstrap["percentiles"] != ["2.5", "50", "97.5"]
    ):
        _fail("EVALUATION_POLICY_INVALID", "bootstrap semantics changed")
    _finite_tree(value, "evaluation_policy")
    return value, sha256_bytes(canonical_json_bytes(value))


def _raw_lock(path: Path) -> tuple[ScorerLock, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactEvaluationError("SCORER_LOCK_UNREADABLE", str(path)) from exc
    lock = ScorerLock.from_dict(value)
    return lock, sha256_bytes(canonical_json_bytes(value))


def _member(role: str, inventory: PredictionInventory, ledger: Ledger) -> dict[str, Any]:
    resolved = resolve_producer(inventory.claim, ledger)
    claim = inventory.claim
    return {
        "role": role,
        "fold_id": inventory.fold.fold_id,
        "producer_run_id": claim.producer_run_id,
        "producer_registration_event_sha256": resolved.registration_event_sha256,
        "producer_terminal_event_sha256": resolved.terminal_event_sha256,
        **claim.registration_evidence(),
        "graph_inventory_sha256": claim.graph_inventory_sha256,
        "artifact_hashes": dict(claim.artifact_hashes),
    }


def _preflight_request(
    request: ExactEvaluationRequest,
) -> tuple[
    Ledger,
    EvaluationManifest,
    ScorerLock,
    dict[str, Any],
    str,
    list[tuple[str, PredictionInventory]],
    list[dict[str, Any]],
]:
    root = request.workspace_root.resolve()
    ledger = Ledger(request.ledger_path, root)
    manifest = load_manifest(request.manifest_path)
    lock, lock_sha = _raw_lock(request.scorer_lock_path)
    policy, policy_sha = load_evaluation_policy(request.evaluation_policy_path)
    if not secrets.compare_digest(manifest.scorer_lock_sha256, lock_sha):
        _fail("SCORER_MANIFEST_MISMATCH", manifest.manifest_sha256)
    expected_folds = tuple(fold.fold_id for fold in manifest.folds)
    inventories: list[tuple[str, PredictionInventory]] = []
    for role, refs in (("baseline", request.baseline_sets), ("candidate", request.candidate_sets)):
        folds = tuple(item.fold_id for item in refs)
        if tuple(sorted(folds)) != expected_folds or len(set(folds)) != len(folds):
            _fail("EVALUATION_MEMBER_SLOTS_INVALID", role)
        for item in sorted(refs, key=lambda value: value.fold_id):
            inventories.append(
                (
                    role,
                    preflight_prediction_set(
                        item.graph_dir,
                        item.producer_manifest_path,
                        request.manifest_path,
                        item.fold_id,
                        ledger,
                    ),
                )
            )
    members = [_member(role, inventory, ledger) for role, inventory in inventories]
    members.sort(key=lambda item: (item["role"], item["fold_id"]))
    events = ledger.read_events()
    state = reconstruct_exact_evaluations(events).get(request.evaluation_run_id)
    if state is None:
        _fail("UNKNOWN_EXACT_EVALUATION", request.evaluation_run_id)
    if state.status is not ExactEvaluationStatus.RUNNING:
        _fail("EXACT_EVALUATION_NOT_STARTED", f"{request.evaluation_run_id}:{state.status}")
    expected_registration = exact_evaluation_registration_payload(
        evaluation_run_id=request.evaluation_run_id,
        scorer_lock_sha256=lock_sha,
        environment_lock_sha256=lock.environment_lock_sha256,
        manifest_sha256=manifest.manifest_sha256,
        evaluation_policy_sha256=policy_sha,
        evidence_kind=state.registered.get("evidence_kind"),
        members=members,
    )
    if canonical_json_bytes(expected_registration) != canonical_json_bytes(state.registered):
        _fail("EXACT_EVALUATION_REGISTRATION_MISMATCH", request.evaluation_run_id)
    return ledger, manifest, lock, policy, policy_sha, inventories, members


def _canonical_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    required = {
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
    }
    if set(summary) != required:
        _fail("OFFICIAL_SUMMARY_INVALID", str(sorted(set(summary) ^ required)))
    result: dict[str, Any] = {}
    for name in sorted(required):
        value = summary[name]
        if name in {"n", "division_tp", "division_fp", "division_fn", "n_adj"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail("OFFICIAL_SUMMARY_INVALID", name)
            result[name] = value
        elif name == "division_jaccard" and isinstance(value, (int, float)) and math.isnan(
            float(value)
        ):
            result[name] = None
        else:
            result[name] = decimal_string(value, name)
    return result


def _score_movie(
    verified: Any,
    prediction: Any,
    truth: Any,
    sample: SampleRecord,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    scored_prediction = prediction.copy()
    scored_truth = truth.copy()
    result = verified.evaluate(
        scored_prediction,
        scored_truth,
        scale=tuple(float(item) for item in sample.scale_zyx_um),
        max_distance=float(verified.lock.raw["constants"]["max_distance_um"]),
    )
    recall = verified.node_recall(scored_prediction, scored_truth)
    organizer_row = verified.per_sample_metrics(
        result, float(sample.estimated_number_of_nodes), recall
    )
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
    if any(value < 0 for value in counts.values()):
        _fail("OFFICIAL_COUNTS_INVALID", sample.sample_id)
    matched_gt_nodes = round(float(recall) * scored_truth.num_nodes())
    node_id_key = verified.tracksdata.DEFAULT_ATTR_KEYS.NODE_ID
    match_key = verified.tracksdata.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID
    match_rows = scored_prediction.node_attrs(
        attr_keys=[node_id_key, match_key]
    ).to_dicts()
    prediction_to_truth = {
        int(row[node_id_key]): int(row[match_key])
        for row in match_rows
        if row[match_key] is not None and int(row[match_key]) >= 0
    }
    division_module = importlib.import_module("tracking_cellmot.division_metrics")
    division_result = division_module.score_divisions(
        prediction.copy(),
        truth.copy(),
        scale=tuple(float(item) for item in sample.scale_zyx_um),
        max_distance=float(verified.lock.raw["constants"]["max_distance_um"]),
    )
    reproduced_division_counts = {
        "division_tp": sum(int(value) for value in division_result.scores.values()),
        "division_fn": len(division_result.scores)
        - sum(int(value) for value in division_result.scores.values()),
        "division_fp": len(division_result.fp_forks),
    }
    if reproduced_division_counts != {
        name: counts[name] for name in ("division_tp", "division_fn", "division_fp")
    }:
        _fail("DIAGNOSTIC_RECONCILIATION_FAILED", f"{sample.sample_id}:division")
    diagnostic_policy = policy["diagnostics"]
    diagnostic = diagnose_movie(
        prediction=graph_data_from_tracksdata(scored_prediction),
        truth=graph_data_from_tracksdata(scored_truth),
        prediction_to_truth=prediction_to_truth,
        official_counts=counts,
        official_adjusted_edge_jaccard=organizer_row["adj_edge_jaccard"],
        estimated_number_of_nodes=sample.estimated_number_of_nodes,
        shape_tzyx=sample.shape_tzyx,
        scale_zyx_um=sample.scale_zyx_um,
        adjustment_alpha=verified.lock.raw["constants"]["adjustment_alpha"],
        displacement_boundaries_um=diagnostic_policy["displacement_boundaries_um"],
        density_boundaries_per_mm3=diagnostic_policy["density_boundaries_per_mm3"],
        density_boundary_rule=diagnostic_policy["density_boundary_rule"],
        division_offsets_frames=diagnostic_policy["division_offsets_frames"],
        no_event_encoding=diagnostic_policy["no_event_encoding"],
        division_scores=division_result.scores,
        division_tp_forks=division_result.tp_forks,
        division_fp_forks=division_result.fp_forks,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No divisions present across any sample in this split*"
        )
        movie_summary = _canonical_summary(verified.summarise([organizer_row]))
    return {
        "organizer_row": organizer_row,
        "official_counts": counts,
        "estimated_number_of_nodes": decimal_string(sample.estimated_number_of_nodes, "estimate"),
        "gt_node_count": scored_truth.num_nodes(),
        "matched_gt_node_count": matched_gt_nodes,
        "node_recall": decimal_string(recall, "node_recall"),
        "node_count_ratio": decimal_string(organizer_row["total_node_ratio"], "node_count_ratio"),
        "edge_jaccard": decimal_string(organizer_row["edge_jaccard"], "edge_jaccard"),
        "division_jaccard": movie_summary["division_jaccard"],
        "adjusted_edge_jaccard": decimal_string(
            organizer_row["adj_edge_jaccard"], "adjusted_edge_jaccard"
        ),
        "score": movie_summary["score"],
        "diagnostic_state": diagnostic,
    }


def aggregate_official_rows(verified: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        _fail("EMPTY_OFFICIAL_GROUP", "official aggregation requires complete movies")
    organizer_rows = [dict(item["organizer_row"]) for item in rows]
    summary = _canonical_summary(verified.summarise(organizer_rows))
    counts = {
        name: sum(int(item["official_counts"][name]) for item in rows)
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
    gt_nodes = sum(int(item["gt_node_count"]) for item in rows)
    matched_nodes = sum(int(item["matched_gt_node_count"]) for item in rows)
    estimate = sum(Decimal(str(item["estimated_number_of_nodes"])) for item in rows)
    weights = [
        int(item["official_counts"]["edge_tp"])
        + int(item["official_counts"]["edge_fp"])
        + int(item["official_counts"]["edge_fn"])
        for item in rows
    ]
    return {
        "movie_count": len(rows),
        "official_counts": counts,
        "estimated_number_of_nodes": decimal_string(estimate, "estimated_number_of_nodes"),
        "gt_node_count": gt_nodes,
        "matched_gt_node_count": matched_nodes,
        "adjusted_edge_weight": sum(weights),
        "edge_jaccard": summary["edge_jaccard"],
        "adjusted_edge_jaccard": summary["adj_edge_jaccard"],
        "division_jaccard": summary["division_jaccard"],
        "organizer_macro_node_recall": summary["node_recall"],
        "node_recall_micro": decimal_string(
            Decimal(matched_nodes) / Decimal(gt_nodes), "node_recall_micro"
        ),
        "score": summary["score"],
    }


def _public_movie(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"organizer_row", "diagnostic_state"}
    }


def _official_projection(verified: Any, movies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_movie = [_public_movie(item) for item in sorted(movies, key=lambda row: row["sample_id"])]
    by_fold: dict[str, Any] = {}
    by_embryo: dict[str, Any] = {}
    for fold_id in sorted({str(item["fold_id"]) for item in movies}):
        selected = [item for item in movies if item["fold_id"] == fold_id]
        by_fold[fold_id] = aggregate_official_rows(verified, selected)
    for embryo_id in sorted({str(item["embryo_id"]) for item in movies}):
        selected = [item for item in movies if item["embryo_id"] == embryo_id]
        by_embryo[embryo_id] = aggregate_official_rows(verified, selected)
    return {
        "pooled": aggregate_official_rows(verified, movies),
        "by_embryo": by_embryo,
        "by_fold": by_fold,
        "by_movie": by_movie,
    }


def _diagnostic_projection(movies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_movie = [
        {
            "sample_id": item["sample_id"],
            "embryo_id": item["embryo_id"],
            "fold_id": item["fold_id"],
            **dict(item["diagnostic_state"]),
        }
        for item in sorted(movies, key=lambda row: row["sample_id"])
    ]
    by_fold: dict[str, Any] = {}
    by_embryo: dict[str, Any] = {}
    for fold_id in sorted({str(item["fold_id"]) for item in movies}):
        by_fold[fold_id] = aggregate_movie_diagnostics(
            [item["diagnostic_state"] for item in movies if item["fold_id"] == fold_id]
        )
    for embryo_id in sorted({str(item["embryo_id"]) for item in movies}):
        by_embryo[embryo_id] = aggregate_movie_diagnostics(
            [item["diagnostic_state"] for item in movies if item["embryo_id"] == embryo_id]
        )
    return {
        "authority": "non_authoritative_diagnostic",
        "organizer_input_eligible": False,
        "pooled": aggregate_movie_diagnostics(
            [item["diagnostic_state"] for item in movies]
        ),
        "by_embryo": by_embryo,
        "by_fold": by_fold,
        "by_movie": by_movie,
    }


def canonical_report_core(
    *,
    registration: Mapping[str, Any],
    authoritative_inventories: Sequence[Mapping[str, Any]],
    official: Mapping[str, Any],
    expected_samples: Sequence[str],
    diagnostics: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": REPORT_SCHEMA,
        "evaluation_run_id": registration["evaluation_run_id"],
        "evidence_kind": registration["evidence_kind"],
        "scorer_lock_sha256": registration["scorer_lock_sha256"],
        "environment_lock_sha256": registration["environment_lock_sha256"],
        "manifest_sha256": registration["manifest_sha256"],
        "evaluation_policy_sha256": registration["evaluation_policy_sha256"],
        "members": list(registration["members"]),
        "authoritative_inventories": list(authoritative_inventories),
        "coverage": {
            "expected_sample_ids": sorted(expected_samples),
            "baseline_sample_ids": sorted(item["sample_id"] for item in official["baseline"]["by_movie"]),
            "candidate_sample_ids": sorted(item["sample_id"] for item in official["candidate"]["by_movie"]),
            "missing": [],
            "extra": [],
            "complete": True,
        },
        "official": dict(official),
        "diagnostics": dict(diagnostics),
        "comparison": dict(comparison),
        "integrity_checks": {
            "producer_ledger_resolution": "passed",
            "aggregate_member_match": "passed",
            "complete_coverage": "passed",
            "native_graph_integrity": "passed",
            "submission_roundtrip": "passed",
            "official_count_parity": "passed",
            "authoritative_prediction_space": "integer-csv-rebuilt-geff",
            "native_prediction_space": "diagnostic_only",
        },
        "promotion_eligible": False,
    }
    validate_exact_core(core)
    return core


def validate_exact_core(core: Any) -> dict[str, Any]:
    if not isinstance(core, dict) or set(core) != _CORE_KEYS:
        _fail("EXACT_REPORT_SCHEMA_INVALID", "unknown or missing core field")
    if core["schema_version"] != REPORT_SCHEMA:
        _fail("EXACT_REPORT_SCHEMA_INVALID", "report schema changed")
    for name in (
        "scorer_lock_sha256",
        "environment_lock_sha256",
        "manifest_sha256",
        "evaluation_policy_sha256",
    ):
        _sha(core[name], name)
    if core["evidence_kind"] not in {
        "synthetic_fixture",
        "official_data_control",
        "model_candidate",
    }:
        _fail("EXACT_REPORT_SCHEMA_INVALID", "evidence kind")
    if not isinstance(core["members"], list) or len(core["members"]) != 4:
        _fail("EXACT_REPORT_SCHEMA_INVALID", "member table")
    coverage = core["coverage"]
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        _fail("EXACT_REPORT_INCOMPLETE", "coverage")
    expected = coverage.get("expected_sample_ids")
    if (
        not isinstance(expected, list)
        or expected != sorted(expected)
        or coverage.get("baseline_sample_ids") != expected
        or coverage.get("candidate_sample_ids") != expected
        or coverage.get("missing") != []
        or coverage.get("extra") != []
    ):
        _fail("EXACT_REPORT_INCOMPLETE", "sample coverage")
    if core.get("promotion_eligible") is not False:
        _fail("EXACT_REPORT_SCHEMA_INVALID", "promotion eligibility attaches only after ledger completion")
    diagnostics = core["diagnostics"]
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("authority") != "non_authoritative_diagnostic"
        or diagnostics.get("organizer_input_eligible") is not False
        or set(diagnostics) != {
            "authority",
            "organizer_input_eligible",
            "baseline",
            "candidate",
        }
    ):
        _fail("EXACT_REPORT_SCHEMA_INVALID", "diagnostic authority boundary")
    comparison = core["comparison"]
    if (
        not isinstance(comparison, dict)
        or comparison.get("schema_version") != "biohub.exact-comparison.v1"
        or comparison.get("direction") != "candidate_minus_baseline"
        or _forbidden_presentation_key(comparison)
    ):
        _fail("EXACT_REPORT_SCHEMA_INVALID", "comparison boundary")
    _finite_tree(core)
    canonical_json_bytes(core)
    return core


def validate_exact_report(
    report: ExactReport,
    *,
    ledger_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    require_completed: bool = True,
) -> ExactReport:
    validate_exact_core(dict(report.core))
    if not isinstance(report.envelope, dict) or set(report.envelope) != _ENVELOPE_KEYS:
        _fail("EXACT_REPORT_SCHEMA_INVALID", "unknown or missing envelope field")
    computed_core = sha256_bytes(canonical_json_bytes(report.core))
    computed_envelope = sha256_bytes(canonical_json_bytes(report.envelope))
    if not secrets.compare_digest(computed_core, report.core_sha256):
        _fail("EXACT_REPORT_HASH_MISMATCH", "core")
    if not secrets.compare_digest(computed_envelope, report.envelope_sha256):
        _fail("EXACT_REPORT_HASH_MISMATCH", "envelope")
    if report.envelope["report_core_sha256"] != report.core_sha256:
        _fail("EXACT_REPORT_HASH_MISMATCH", "envelope core reference")
    _finite_tree(report.envelope)
    if require_completed:
        if ledger_path is None or workspace_root is None:
            _fail("EXACT_LEDGER_REQUIRED", "completed report validation requires a ledger")
        ledger = Ledger(Path(ledger_path), Path(workspace_root))
        state = reconstruct_exact_evaluations(ledger.read_events()).get(
            str(report.core["evaluation_run_id"])
        )
        if state is None or state.status is not ExactEvaluationStatus.COMPLETED or state.terminal is None:
            _fail("EXACT_EVALUATION_NOT_COMPLETED", str(report.core["evaluation_run_id"]))
        if state.terminal.get("report_core_sha256") != report.core_sha256:
            _fail("EXACT_REPORT_HASH_MISMATCH", "ledger core")
        if state.terminal.get("envelope_sha256") != report.envelope_sha256:
            _fail("EXACT_REPORT_HASH_MISMATCH", "ledger envelope")
        if canonical_json_bytes(state.registered["members"]) != canonical_json_bytes(
            report.core["members"]
        ):
            _fail("EXACT_EVALUATION_REGISTRATION_MISMATCH", "ledger members")
    return report


def _failure_code(exc: Exception) -> str:
    value = getattr(exc, "reason_code", None)
    return str(value) if value else type(exc).__name__.upper()


def evaluate_exact(request: ExactEvaluationRequest) -> ExactReport:
    """Run the complete fail-closed reciprocal evaluation and attach it once."""

    started_at = time.perf_counter()
    output = request.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable exact report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e-", dir=output.parent))
    published = False
    ledger: Ledger | None = None
    tracemalloc.start()
    try:
        (
            ledger,
            manifest,
            lock,
            policy,
            policy_sha,
            inventories,
            members,
        ) = _preflight_request(request)
        verified = verify_scorer_lock(
            request.scorer_lock_path,
            request.scorer_checkout,
            tracksdata_checkout=request.tracksdata_checkout,
        )
        if verified.lock_sha256 != manifest.scorer_lock_sha256:
            _fail("SCORER_MANIFEST_MISMATCH", manifest.manifest_sha256)
        samples = {item.sample_id: item for item in manifest.samples}
        truth_root = request.truth_dir.resolve(strict=True)
        roundtrips: list[tuple[str, PredictionInventory, RoundTripEvidence, Path]] = []
        authoritative: list[dict[str, str]] = []
        for index, (role, inventory) in enumerate(inventories):
            # Keep the Windows path comfortably below legacy path limits; the
            # full immutable role/fold identity lives in evidence, not directory names.
            target = staging / f"r{index}"
            evidence = roundtrip_prediction_inventory(
                inventory, verified, truth_root, target
            )
            authoritative.append(
                {
                    "role": role,
                    "fold_id": inventory.fold.fold_id,
                    "producer_run_id": inventory.claim.producer_run_id,
                    "submission_graph_inventory_sha256": evidence.submission_graph_inventory_sha256,
                    "roundtrip_evidence_sha256": evidence.evidence_sha256,
                    "csv_sha256": evidence.csv_sha256,
                }
            )
            roundtrips.append((role, inventory, evidence, target))
        authoritative.sort(key=lambda item: (item["role"], item["fold_id"]))
        movies: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
        for role, inventory, evidence, target in roundtrips:
            roundtrip_counts = {
                item["sample_id"]: item["official_counts"] for item in evidence.per_movie
            }
            member = next(
                item
                for item in members
                if item["role"] == role and item["fold_id"] == inventory.fold.fold_id
            )
            for sample_id in inventory.fold.evaluation_membership:
                sample = samples[sample_id]
                prediction_path = target / "graphs" / f"{sample_id}.geff"
                truth_path = (truth_root / sample.truth_relpath).resolve(strict=True)
                try:
                    truth_path.relative_to(truth_root)
                except ValueError as exc:
                    raise ExactEvaluationError("TRUTH_PATH_ESCAPE", sample.truth_relpath) from exc
                if artifact_tree_sha256(truth_path) != sample.geff_tree_sha256:
                    _fail("TRUTH_HASH_MISMATCH", sample_id)
                row = _score_movie(
                    verified,
                    load_geff_graph(prediction_path, verified),
                    load_geff_graph(truth_path, verified),
                    sample,
                    policy,
                )
                if row["official_counts"] != roundtrip_counts[sample_id]:
                    _fail("ROUNDTRIP_COUNT_MISMATCH", sample_id)
                row.update(
                    {
                        "sample_id": sample_id,
                        "embryo_id": sample.embryo_id,
                        "fold_id": inventory.fold.fold_id,
                        "producer": member,
                    }
                )
                movies[role].append(row)
        official = {
            role: _official_projection(verified, movies[role])
            for role in ("baseline", "candidate")
        }
        diagnostics = {
            "authority": "non_authoritative_diagnostic",
            "organizer_input_eligible": False,
            **{
                role: _diagnostic_projection(movies[role])
                for role in ("baseline", "candidate")
            },
        }
        state = reconstruct_exact_evaluations(ledger.read_events())[request.evaluation_run_id]
        comparison = build_paired_comparison(
            registration=state.registered,
            aggregate_status=state.status.value,
            members=members,
            authoritative_inventories=authoritative,
            baseline_movies=movies["baseline"],
            candidate_movies=movies["candidate"],
            expected_sample_ids=[item.sample_id for item in manifest.samples],
            policy=policy,
            policy_sha256=policy_sha,
            aggregate=lambda rows: aggregate_official_rows(verified, rows),
        )
        core = canonical_report_core(
            registration=state.registered,
            authoritative_inventories=authoritative,
            official=official,
            expected_samples=[item.sample_id for item in manifest.samples],
            diagnostics=diagnostics,
            comparison=comparison,
        )
        core_sha = sha256_bytes(canonical_json_bytes(core))
        _, peak_bytes = tracemalloc.get_traced_memory()
        envelope = {
            "schema_version": "biohub.exact-report-envelope.v1",
            "report_core_sha256": core_sha,
            "created_at": utc_now().isoformat().replace("+00:00", "Z"),
            "runtime_seconds": decimal_string(time.perf_counter() - started_at, "runtime"),
            "peak_memory_bytes": int(peak_bytes),
            "presentation_metadata": dict(request.presentation_metadata or {}),
        }
        _finite_tree(envelope)
        envelope_sha = sha256_bytes(canonical_json_bytes(envelope))
        core_path = staging / "exact-report-core.json"
        envelope_path = staging / "exact-report-envelope.json"
        atomic_write_json(core_path, core)
        atomic_write_json(envelope_path, envelope)
        os.rename(staging, output)
        published = True
        core_path = output / core_path.name
        envelope_path = output / envelope_path.name
        completion = exact_evaluation_completed_payload(
            evaluation_run_id=request.evaluation_run_id,
            members=members,
            report_core_sha256=core_sha,
            envelope_sha256=envelope_sha,
            artifact_hashes={
                "report_core_file": sha256_file(core_path),
                "report_envelope_file": sha256_file(envelope_path),
            },
            authoritative_inventories=authoritative,
        )
        ledger.append(
            ExperimentEvent.create(
                request.evaluation_run_id,
                EventType.EXACT_EVALUATION_COMPLETED,
                completion,
            )
        )
        report = ExactReport(
            core,
            envelope,
            core_sha,
            envelope_sha,
            core_path,
            envelope_path,
        )
        return validate_exact_report(
            report,
            ledger_path=request.ledger_path,
            workspace_root=request.workspace_root,
        )
    except Exception as exc:
        if published and output.exists():
            shutil.rmtree(output)
        elif staging.exists():
            shutil.rmtree(staging)
        if ledger is not None:
            try:
                state = reconstruct_exact_evaluations(ledger.read_events()).get(
                    request.evaluation_run_id
                )
                if state is not None and state.status is ExactEvaluationStatus.RUNNING:
                    ledger.append(
                        ExperimentEvent.create(
                            request.evaluation_run_id,
                            EventType.EXACT_EVALUATION_FAILED,
                            exact_evaluation_failed_payload(
                                evaluation_run_id=request.evaluation_run_id,
                                reason_code=_failure_code(exc),
                                detail=str(exc),
                            ),
                        )
                    )
            except Exception:
                pass
        raise
    finally:
        tracemalloc.stop()
        if staging.exists():
            shutil.rmtree(staging)
