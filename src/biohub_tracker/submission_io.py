from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from .graphs import (
    PredictionInventory,
    artifact_tree_sha256,
    graph_data_from_tracksdata,
    load_geff_graph,
    validate_graph_data,
)
from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file


SUBMISSION_HEADER = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")


class RoundTripError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> None:
    raise RoundTripError(reason_code, detail)


@dataclass(frozen=True)
class CsvValidationResult:
    dataset_ids: tuple[str, ...]
    row_count: int
    node_counts: tuple[tuple[str, int], ...]
    edge_counts: tuple[tuple[str, int], ...]
    csv_sha256: str


@dataclass(frozen=True)
class RoundTripEvidence:
    producer_run_id: str
    fold_id: str
    manifest_sha256: str
    source_graph_inventory_sha256: str
    submission_graph_inventory_sha256: str
    csv_sha256: str
    source_space: str
    authoritative_space: str
    semantic_parity: bool
    official_counts_parity: bool
    source_lineage: Mapping[str, Any]
    graphs: tuple[Mapping[str, Any], ...]
    per_movie: tuple[Mapping[str, Any], ...]
    aggregate: Mapping[str, Any]
    evidence_sha256: str
    schema_version: int = 1

    def semantic_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("evidence_sha256", None)
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_pinned_script(verified: Any, filename: str) -> ModuleType:
    path = (verified.checkout / "scripts" / filename).resolve(strict=True)
    try:
        path.relative_to(verified.checkout.resolve(strict=True))
    except ValueError as exc:
        raise RoundTripError("CONVERTER_PATH_INVALID", str(path)) from exc
    expected = dict(verified.lock.critical_files).get(f"scripts/{filename}")
    if expected is None or sha256_file(path) != expected:
        _fail("CONVERTER_HASH_MISMATCH", filename)
    name = f"_biohub_pinned_{path.stem}_{expected[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail("CONVERTER_IMPORT_FAILED", filename)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RoundTripError("CONVERTER_IMPORT_FAILED", filename) from exc
    return module


def _parse_integer(value: str, field: str, row_id: int) -> int:
    if _INTEGER.fullmatch(value) is None:
        _fail("CSV_NONINTEGER_FIELD", f"row={row_id}, field={field}, value={value!r}")
    return int(value)


def validate_submission_csv(
    path: str | Path,
    expected_datasets: Sequence[str],
) -> CsvValidationResult:
    expected = tuple(expected_datasets)
    if not expected or expected != tuple(sorted(expected)) or len(set(expected)) != len(expected):
        _fail("CSV_EXPECTED_COVERAGE_INVALID", str(expected))
    node_ids: dict[str, set[int]] = {name: set() for name in expected}
    edge_pairs: dict[str, set[tuple[int, int]]] = {name: set() for name in expected}
    node_counts = {name: 0 for name in expected}
    edge_counts = {name: 0 for name in expected}
    dataset_order = {name: index for index, name in enumerate(expected)}
    previous_dataset = -1
    seen_edge: set[str] = set()
    last_node_id: dict[str, int | None] = {name: None for name in expected}
    last_edge: dict[str, tuple[int, int] | None] = {name: None for name in expected}
    pending_edges: list[tuple[int, str, int, int]] = []
    try:
        handle = Path(path).open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise RoundTripError("CSV_UNREADABLE", str(path)) from exc
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUBMISSION_HEADER:
            _fail("CSV_HEADER_MISMATCH", str(reader.fieldnames))
        rows = list(reader)
    if not rows:
        _fail("CSV_EMPTY", str(path))
    for row_id, row in enumerate(rows):
        if set(row) != set(SUBMISSION_HEADER) or None in row:
            _fail("CSV_SCHEMA_DRIFT", f"row={row_id}")
        if _parse_integer(row["id"], "id", row_id) != row_id:
            _fail("CSV_ROW_ID_INVALID", f"row={row_id}")
        dataset = row["dataset"]
        if dataset not in dataset_order:
            _fail("CSV_COVERAGE_MISMATCH", dataset)
        current_dataset = dataset_order[dataset]
        if current_dataset < previous_dataset:
            _fail("CSV_DATASET_ORDER_INVALID", f"row={row_id}")
        previous_dataset = current_dataset
        row_type = row["row_type"]
        values = {
            field: _parse_integer(row[field], field, row_id)
            for field in SUBMISSION_HEADER[3:]
        }
        if row_type == "node":
            if dataset in seen_edge:
                _fail("CSV_ROW_ORDER_INVALID", f"row={row_id}")
            if values["source_id"] != -1 or values["target_id"] != -1:
                _fail("CSV_PLACEHOLDER_INVALID", f"row={row_id}")
            if any(values[field] < 0 for field in ("node_id", "t", "z", "y", "x")):
                _fail("CSV_SENTINEL_INVALID", f"row={row_id}")
            if values["node_id"] in node_ids[dataset]:
                _fail("CSV_DUPLICATE_NODE_ID", f"{dataset}:{values['node_id']}")
            if last_node_id[dataset] is not None and values["node_id"] < last_node_id[dataset]:
                _fail("CSV_NODE_ORDER_INVALID", f"row={row_id}")
            node_ids[dataset].add(values["node_id"])
            last_node_id[dataset] = values["node_id"]
            node_counts[dataset] += 1
        elif row_type == "edge":
            seen_edge.add(dataset)
            if any(values[field] != -1 for field in ("node_id", "t", "z", "y", "x")):
                _fail("CSV_PLACEHOLDER_INVALID", f"row={row_id}")
            source, target = values["source_id"], values["target_id"]
            if source < 0 or target < 0:
                _fail("CSV_SENTINEL_INVALID", f"row={row_id}")
            pair = (source, target)
            if pair in edge_pairs[dataset]:
                _fail("CSV_DUPLICATE_EDGE", f"{dataset}:{source}->{target}")
            if last_edge[dataset] is not None and pair < last_edge[dataset]:
                _fail("CSV_EDGE_ORDER_INVALID", f"row={row_id}")
            edge_pairs[dataset].add(pair)
            last_edge[dataset] = pair
            edge_counts[dataset] += 1
            pending_edges.append((row_id, dataset, source, target))
        else:
            _fail("CSV_ROW_TYPE_INVALID", f"row={row_id}, type={row_type!r}")
    covered = tuple(name for name in expected if node_counts[name] or edge_counts[name])
    if covered != expected:
        _fail("CSV_COVERAGE_MISMATCH", f"missing={sorted(set(expected)-set(covered))}")
    for row_id, dataset, source, target in pending_edges:
        if source not in node_ids[dataset] or target not in node_ids[dataset]:
            _fail("CSV_DANGLING_EDGE", f"row={row_id}, {dataset}:{source}->{target}")
    return CsvValidationResult(
        dataset_ids=expected,
        row_count=len(rows),
        node_counts=tuple((name, node_counts[name]) for name in expected),
        edge_counts=tuple((name, edge_counts[name]) for name in expected),
        csv_sha256=sha256_file(path),
    )


def _semantic_graph(graph: Any) -> dict[str, Any]:
    data = graph_data_from_tracksdata(graph)
    nodes = sorted(
        (
            int(node.node_id),
            int(node.t),
            int(node.z),
            int(node.y),
            int(node.x),
        )
        for node in data.nodes
    )
    coordinate_groups: dict[tuple[int, int, int, int], list[int]] = {}
    for node_id, t, z, y, x in nodes:
        coordinate_groups.setdefault((t, z, y, x), []).append(node_id)
    labels: dict[int, tuple[int, int, int, int, int]] = {}
    node_labels: list[tuple[int, int, int, int, int]] = []
    for coordinate in sorted(coordinate_groups):
        for occurrence, node_id in enumerate(sorted(coordinate_groups[coordinate])):
            label = (*coordinate, occurrence)
            labels[node_id] = label
            node_labels.append(label)
    edges = sorted((labels[int(source)], labels[int(target)]) for source, target in data.edges)
    return {"nodes": node_labels, "edges": edges}


def _official_evidence(verified: Any, prediction: Any, truth: Any, sample: Any) -> tuple[dict[str, int], dict[str, Any]]:
    scored_prediction = prediction.copy()
    scored_truth = truth.copy()
    result = verified.evaluate(
        scored_prediction,
        scored_truth,
        scale=tuple(float(item) for item in sample.scale_zyx_um),
        max_distance=float(verified.lock.raw["constants"]["max_distance_um"]),
    )
    fields = (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
        "num_pred_nodes",
    )
    counts = {name: int(getattr(result, name)) for name in fields}
    recall = verified.node_recall(scored_prediction, scored_truth)
    row = verified.per_sample_metrics(result, float(sample.estimated_number_of_nodes), recall)
    return counts, row


def _same_exact(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _lineage(claim: Any) -> dict[str, Any]:
    return {
        "producer_run_id": claim.producer_run_id,
        **claim.registration_evidence(),
        "graph_inventory_sha256": claim.graph_inventory_sha256,
        "artifact_hashes": dict(claim.artifact_hashes),
    }


def roundtrip_prediction_inventory(
    inventory: PredictionInventory,
    verified: Any,
    truth_root: str | Path,
    output_dir: str | Path,
) -> RoundTripEvidence:
    """Project a resolved complete native set and publish only verified rebuilt graphs."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable round-trip output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    truth_base = Path(truth_root).resolve(strict=True)
    samples = {sample.sample_id: sample for sample in inventory.manifest.samples}
    expected = tuple(inventory.fold.evaluation_membership)
    geff_to_csv = _load_pinned_script(verified, "geffs_to_csv.py")
    csv_to_geff = _load_pinned_script(verified, "csv_to_geffs.py")
    polars = __import__("polars")
    native_graphs: dict[str, Any] = {}
    frames: list[Any] = []
    for sample_id, path in inventory.graph_paths:
        sample = samples[sample_id]
        graph = load_geff_graph(path, verified)
        validate_graph_data(
            graph_data_from_tracksdata(graph), sample, mode="native", graph_sha256=artifact_tree_sha256(path)
        )
        native_graphs[sample_id] = graph
        frame = geff_to_csv.graph_to_rows(graph, sample_id)
        node_rows = frame.filter(polars.col("row_type") == "node").sort("node_id")
        edge_rows = frame.filter(polars.col("row_type") == "edge").sort("source_id", "target_id")
        frames.append(polars.concat([node_rows, edge_rows]))
    table = polars.concat(frames).with_row_index("id")
    if tuple(table.columns) != SUBMISSION_HEADER:
        _fail("CSV_HEADER_MISMATCH", str(table.columns))

    staging = Path(tempfile.mkdtemp(prefix=".biohub-roundtrip-", dir=output.parent))
    try:
        csv_path = staging / "submission.csv"
        table.write_csv(csv_path)
        csv_result = validate_submission_csv(csv_path, expected)
        parsed = polars.read_csv(csv_path, columns=list(SUBMISSION_HEADER))
        graph_dir = staging / "graphs"
        graph_dir.mkdir()
        graph_records: list[dict[str, Any]] = []
        per_movie: list[dict[str, Any]] = []
        direct_rows: list[dict[str, Any]] = []
        rebuilt_rows: list[dict[str, Any]] = []
        for sample_id in expected:
            group = parsed.filter(polars.col("dataset") == sample_id)
            node_rows = group.filter(polars.col("row_type") == "node")
            edge_rows = group.filter(polars.col("row_type") == "edge")
            direct = csv_to_geff.build_graph_from_rows(node_rows, edge_rows)
            rebuilt_path = graph_dir / f"{sample_id}.geff"
            direct.to_geff(rebuilt_path)
            rebuilt = load_geff_graph(rebuilt_path, verified)
            sample = samples[sample_id]
            validation = validate_graph_data(
                graph_data_from_tracksdata(rebuilt), sample, mode="submission"
            )
            if not _same_exact(_semantic_graph(direct), _semantic_graph(rebuilt)):
                _fail("SEMANTIC_TOPOLOGY_DRIFT", sample_id)
            truth_path = (truth_base / sample.truth_relpath).resolve(strict=True)
            try:
                truth_path.relative_to(truth_base)
            except ValueError as exc:
                raise RoundTripError("TRUTH_PATH_ESCAPE", sample.truth_relpath) from exc
            if artifact_tree_sha256(truth_path) != sample.geff_tree_sha256:
                _fail("TRUTH_HASH_MISMATCH", sample_id)
            truth = load_geff_graph(truth_path, verified)
            direct_counts, direct_row = _official_evidence(verified, direct, truth, sample)
            rebuilt_counts, rebuilt_row = _official_evidence(verified, rebuilt, truth, sample)
            if direct_counts != rebuilt_counts or not _same_exact(direct_row, rebuilt_row):
                _fail("OFFICIAL_METRIC_DRIFT", sample_id)
            direct_rows.append(direct_row)
            rebuilt_rows.append(rebuilt_row)
            digest = artifact_tree_sha256(rebuilt_path)
            graph_records.append(
                {
                    "sample_id": sample_id,
                    "path": f"graphs/{sample_id}.geff",
                    "sha256": digest,
                    "node_count": validation.node_count,
                    "edge_count": validation.edge_count,
                }
            )
            per_movie.append(
                {
                    "sample_id": sample_id,
                    "official_counts": direct_counts,
                    "row_sha256": sha256_bytes(canonical_json_bytes(direct_row)),
                }
            )
        direct_summary = verified.summarise(direct_rows)
        rebuilt_summary = verified.summarise(rebuilt_rows)
        if not _same_exact(direct_summary, rebuilt_summary):
            _fail("OFFICIAL_AGGREGATE_DRIFT", inventory.fold.fold_id)
        submission_inventory_sha256 = sha256_bytes(canonical_json_bytes(graph_records))
        evidence = RoundTripEvidence(
            producer_run_id=inventory.claim.producer_run_id,
            fold_id=inventory.fold.fold_id,
            manifest_sha256=inventory.manifest.manifest_sha256,
            source_graph_inventory_sha256=inventory.claim.graph_inventory_sha256,
            submission_graph_inventory_sha256=submission_inventory_sha256,
            csv_sha256=csv_result.csv_sha256,
            source_space="native-nonauthoritative",
            authoritative_space="integer-csv-rebuilt-geff",
            semantic_parity=True,
            official_counts_parity=True,
            source_lineage=_lineage(inventory.claim),
            graphs=tuple(graph_records),
            per_movie=tuple(per_movie),
            aggregate={"summary_sha256": sha256_bytes(canonical_json_bytes(direct_summary))},
            evidence_sha256="",
        )
        evidence = RoundTripEvidence(
            **{**evidence.__dict__, "evidence_sha256": sha256_bytes(canonical_json_bytes(evidence.semantic_dict()))}
        )
        atomic_write_json(staging / "submission-space-inventory.json", evidence.to_dict())
        os.rename(staging, output)
        return evidence
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
