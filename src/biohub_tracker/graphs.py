from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .evidence import EvidenceError, PredictionSetClaim, load_prediction_set, resolve_producer
from .io import sha256_bytes, sha256_file
from .ledger import Ledger
from .manifests import EvaluationManifest, FoldRecord, SampleRecord, load_manifest


class GraphValidationError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> None:
    raise GraphValidationError(reason_code, detail)


def artifact_tree_sha256(directory: str | Path) -> str:
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        _fail("GRAPH_UNREADABLE", str(directory))
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        _fail("GRAPH_UNREADABLE", f"empty graph artifact: {directory}")
    value = bytearray()
    for path in files:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise GraphValidationError("GRAPH_PATH_ESCAPE", str(path)) from exc
        value.extend(path.relative_to(root).as_posix().encode("utf-8"))
        value.extend(b"\0")
        value.extend(bytes.fromhex(sha256_file(resolved)))
        value.extend(b"\0")
    return sha256_bytes(bytes(value))


@dataclass(frozen=True)
class GraphNode:
    node_id: Any
    t: Any
    z: Any
    y: Any
    x: Any


@dataclass(frozen=True)
class GraphData:
    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class GraphValidationResult:
    sample_id: str
    mode: str
    node_count: int
    edge_count: int
    graph_sha256: str | None = None


@dataclass(frozen=True)
class PredictionInventory:
    pred_dir: Path
    claim: PredictionSetClaim
    manifest: EvaluationManifest
    fold: FoldRecord
    graph_paths: tuple[tuple[str, Path], ...]


@dataclass(frozen=True)
class PredictionValidationResult:
    producer_run_id: str
    fold_id: str
    manifest_sha256: str
    graph_inventory_sha256: str
    mode: str
    graphs: tuple[GraphValidationResult, ...]


def _is_integral(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and float(value).is_integer()


def _coordinate(value: Any, *, mode: str, sample_id: str, node_id: Any, axis: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail("NONFINITE_COORDINATE", f"{sample_id}:{node_id}:{axis}")
    if mode == "submission" and not _is_integral(value):
        _fail("NONINTEGRAL_COORDINATE", f"{sample_id}:{node_id}:{axis}={value}")
    return float(value)


def validate_graph_data(
    data: GraphData,
    sample: SampleRecord,
    *,
    mode: str,
    graph_sha256: str | None = None,
) -> GraphValidationResult:
    if mode not in {"native", "submission"}:
        _fail("GRAPH_MODE_INVALID", mode)
    nodes = sorted(data.nodes, key=lambda item: (str(type(item.node_id)), str(item.node_id)))
    node_values: dict[int, GraphNode] = {}
    t_bound, z_bound, y_bound, x_bound = sample.shape_tzyx
    for node in nodes:
        if not _is_integral(node.node_id):
            _fail("NONINTEGRAL_NODE_ID", f"{sample.sample_id}:{node.node_id}")
        node_id = int(node.node_id)
        if node_id < 0:
            _fail("SENTINEL_VALUE", f"{sample.sample_id}:node_id={node_id}")
        if node_id in node_values:
            _fail("DUPLICATE_NODE_ID", f"{sample.sample_id}:{node_id}")
        if not _is_integral(node.t):
            _fail("NONINTEGRAL_TIME", f"{sample.sample_id}:{node_id}:t={node.t}")
        time = int(node.t)
        if time < 0:
            _fail("SENTINEL_VALUE", f"{sample.sample_id}:{node_id}:t={time}")
        if time >= t_bound:
            _fail("OUT_OF_BOUNDS", f"{sample.sample_id}:{node_id}:t={time}")
        coordinates = (
            ("z", _coordinate(node.z, mode=mode, sample_id=sample.sample_id, node_id=node_id, axis="z"), z_bound),
            ("y", _coordinate(node.y, mode=mode, sample_id=sample.sample_id, node_id=node_id, axis="y"), y_bound),
            ("x", _coordinate(node.x, mode=mode, sample_id=sample.sample_id, node_id=node_id, axis="x"), x_bound),
        )
        for axis, coordinate, bound in coordinates:
            if coordinate < 0:
                _fail("SENTINEL_VALUE", f"{sample.sample_id}:{node_id}:{axis}={coordinate}")
            if coordinate >= bound:
                _fail("OUT_OF_BOUNDS", f"{sample.sample_id}:{node_id}:{axis}={coordinate}")
        node_values[node_id] = GraphNode(node_id, time, node.z, node.y, node.x)

    edges: list[tuple[int, int]] = []
    for raw_source, raw_target in data.edges:
        if not _is_integral(raw_source) or not _is_integral(raw_target):
            _fail("NONINTEGRAL_EDGE_ENDPOINT", f"{sample.sample_id}:{raw_source}->{raw_target}")
        source, target = int(raw_source), int(raw_target)
        edges.append((source, target))
    edges.sort()
    if len(set(edges)) != len(edges):
        _fail("DUPLICATE_EDGE", sample.sample_id)
    indegree = {node_id: 0 for node_id in node_values}
    outdegree = {node_id: 0 for node_id in node_values}
    adjacency = {node_id: [] for node_id in node_values}
    for source, target in edges:
        if source not in node_values or target not in node_values:
            _fail("DANGLING_ENDPOINT", f"{sample.sample_id}:{source}->{target}")
        if source == target:
            _fail("SELF_LOOP", f"{sample.sample_id}:{source}")
        indegree[target] += 1
        outdegree[source] += 1
        adjacency[source].append(target)
    merged = sorted(node_id for node_id, count in indegree.items() if count > 1)
    if merged:
        _fail("MERGED_DAUGHTER", f"{sample.sample_id}:{merged}")
    hubs = sorted(node_id for node_id, count in outdegree.items() if count > 2)
    if hubs:
        _fail("FAKE_FORK", f"{sample.sample_id}:{hubs}")

    remaining = dict(indegree)
    ready = sorted(node_id for node_id, degree in remaining.items() if degree == 0)
    visited = 0
    while ready:
        node_id = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[node_id]):
            remaining[target] -= 1
            if remaining[target] == 0:
                ready.append(target)
                ready.sort()
    if visited != len(node_values):
        _fail("CYCLE", sample.sample_id)
    for source, target in edges:
        if int(node_values[target].t) != int(node_values[source].t) + 1:
            _fail("FRAME_STEP_INVALID", f"{sample.sample_id}:{source}->{target}")
    return GraphValidationResult(
        sample_id=sample.sample_id,
        mode=mode,
        node_count=len(node_values),
        edge_count=len(edges),
        graph_sha256=graph_sha256,
    )


def graph_data_from_tracksdata(graph: Any) -> GraphData:
    try:
        node_rows = graph.node_attrs().select("node_id", "t", "z", "y", "x").to_dicts()
        edge_rows = graph.edge_attrs().select("source_id", "target_id").to_dicts()
    except Exception as exc:
        raise GraphValidationError("GRAPH_SCHEMA_INVALID", str(exc)) from exc
    return GraphData(
        nodes=tuple(
            GraphNode(row["node_id"], row["t"], row["z"], row["y"], row["x"])
            for row in node_rows
        ),
        edges=tuple((row["source_id"], row["target_id"]) for row in edge_rows),
    )


def load_geff_graph(path: Path, verified: Any) -> Any:
    try:
        result = verified.tracksdata.graph.IndexedRXGraph.from_geff(path)
    except Exception as exc:
        raise GraphValidationError("GRAPH_UNREADABLE", str(path)) from exc
    return result[0] if isinstance(result, tuple) else result


def preflight_prediction_set(
    pred_dir: str | Path,
    producer_manifest: str | Path,
    manifest_path: str | Path,
    fold_id: str,
    ledger: Ledger,
) -> PredictionInventory:
    """Complete every cheap provenance/coverage check before scientific imports."""

    claim = load_prediction_set(producer_manifest)
    resolve_producer(claim, ledger)
    manifest = load_manifest(manifest_path)
    if claim.manifest_sha256 != manifest.manifest_sha256:
        _fail("MANIFEST_MISMATCH", claim.manifest_sha256)
    folds = {fold.fold_id: fold for fold in manifest.folds}
    fold = folds.get(fold_id)
    if fold is None or claim.fold_id != fold_id:
        _fail("FOLD_MISMATCH", f"requested={fold_id}, claimed={claim.fold_id}")
    expected_memberships = (
        fold.train_membership_sha256,
        fold.calibration_membership_sha256,
        fold.evaluation_membership_sha256,
    )
    claimed_memberships = (
        claim.train_membership_sha256,
        claim.calibration_membership_sha256,
        claim.evaluation_membership_sha256,
    )
    if expected_memberships != claimed_memberships:
        _fail("MEMBERSHIP_MISMATCH", fold_id)
    if not set(fold.calibration_membership) <= set(fold.train_membership):
        _fail("CALIBRATION_LEAKAGE", fold_id)
    if any(item.producer_run_id != claim.producer_run_id for item in claim.graphs):
        _fail("MIXED_PRODUCER", claim.producer_run_id)
    if any(item.fold_id != claim.fold_id for item in claim.graphs):
        _fail("MIXED_FOLD", claim.fold_id)
    expected = set(fold.evaluation_membership)
    claimed = {item.sample_id for item in claim.graphs}
    if claimed != expected:
        _fail(
            "GRAPH_COVERAGE_MISMATCH",
            f"missing={sorted(expected-claimed)}, extra={sorted(claimed-expected)}",
        )

    root = Path(pred_dir).resolve(strict=True)
    if not root.is_dir():
        _fail("PREDICTION_DIRECTORY_INVALID", str(pred_dir))
    discovered = {path.stem for path in root.glob("*.geff") if path.is_dir()}
    if discovered != expected:
        _fail(
            "GRAPH_COVERAGE_MISMATCH",
            f"missing={sorted(expected-discovered)}, extra={sorted(discovered-expected)}",
        )
    paths: list[tuple[str, Path]] = []
    for item in claim.graphs:
        if item.path != f"{item.sample_id}.geff":
            _fail("GRAPH_PATH_MISMATCH", f"{item.sample_id}:{item.path}")
        path = (root / item.path).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GraphValidationError("GRAPH_PATH_ESCAPE", item.path) from exc
        digest = artifact_tree_sha256(path)
        if digest != item.sha256:
            _fail("STALE_GRAPH_HASH", item.sample_id)
        paths.append((item.sample_id, path))
    return PredictionInventory(root, claim, manifest, fold, tuple(paths))


def validate_prediction_inventory(
    inventory: PredictionInventory,
    *,
    mode: str,
    graph_loader: Callable[[Path], Any],
) -> PredictionValidationResult:
    samples = {sample.sample_id: sample for sample in inventory.manifest.samples}
    results: list[GraphValidationResult] = []
    hashes = {item.sample_id: item.sha256 for item in inventory.claim.graphs}
    for sample_id, path in inventory.graph_paths:
        graph = graph_loader(path)
        results.append(
            validate_graph_data(
                graph_data_from_tracksdata(graph),
                samples[sample_id],
                mode=mode,
                graph_sha256=hashes[sample_id],
            )
        )
    return PredictionValidationResult(
        producer_run_id=inventory.claim.producer_run_id,
        fold_id=inventory.fold.fold_id,
        manifest_sha256=inventory.manifest.manifest_sha256,
        graph_inventory_sha256=inventory.claim.graph_inventory_sha256,
        mode=mode,
        graphs=tuple(results),
    )


def validate_prediction_set(
    pred_dir: str | Path,
    producer_manifest: str | Path,
    manifest_path: str | Path,
    fold_id: str,
    ledger: Ledger,
    *,
    mode: str,
    graph_loader: Callable[[Path], Any],
) -> PredictionValidationResult:
    inventory = preflight_prediction_set(
        pred_dir, producer_manifest, manifest_path, fold_id, ledger
    )
    return validate_prediction_inventory(inventory, mode=mode, graph_loader=graph_loader)
