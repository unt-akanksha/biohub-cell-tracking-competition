from __future__ import annotations

import json
import math
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file, utc_now


SCHEMA_VERSION = 1
OFFICIAL_EMBRYOS = ("44b6", "6bba")
OFFICIAL_AXES = ("t", "z", "y", "x")
_SAMPLE_ID = re.compile(r"^(44b6|6bba)_(.+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> None:
    raise ManifestError(reason_code, detail)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("INVALID_METADATA", f"{name} must be an object")
    return value


def _positive_ints(value: Any, name: str, length: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        _fail("INVALID_METADATA", f"{name} must contain {length} positive integers")
    return tuple(value)


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("INVALID_METADATA", f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        _fail("INVALID_METADATA", f"{name} must be finite and positive")
    return result


def _decimal_text(value: Any) -> str:
    result = _finite_positive(value, "decimal value")
    return format(result, ".15g")


def _json(path: Path, reason: str = "INVALID_METADATA") -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(reason, str(path)) from exc
    if not isinstance(value, Mapping):
        _fail(reason, f"{path} must contain an object")
    return value


def _contained(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ManifestError("PATH_ESCAPE", str(path)) from exc
    return resolved


def _tree_sha256(directory: Path) -> str:
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not files:
        _fail("INVALID_METADATA", f"artifact tree is empty: {directory}")
    digest_input = bytearray()
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest_input.extend(relative)
        digest_input.extend(b"\0")
        digest_input.extend(bytes.fromhex(sha256_file(path)))
        digest_input.extend(b"\0")
    return sha256_bytes(bytes(digest_input))


def _array_shape(root: Path, relative: str, *, dimensions: int) -> tuple[int, ...]:
    metadata = _json(root / relative / "zarr.json")
    return _positive_ints(metadata.get("shape"), f"{relative}.shape", dimensions)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    embryo_id: str
    field_of_view_id: str
    image_relpath: str
    truth_relpath: str
    shape_tzyx: tuple[int, int, int, int]
    dtype: str
    chunks_tzyx: tuple[int, int, int, int]
    scale_zyx_um: tuple[str, str, str]
    estimated_number_of_nodes: str
    gt_node_count: int
    gt_edge_count: int
    image_metadata_sha256: str
    geff_tree_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("shape_tzyx", "chunks_tzyx", "scale_zyx_um"):
            value[name] = list(value[name])
        return value

    @classmethod
    def from_dict(cls, value: Any) -> "SampleRecord":
        data = dict(_mapping(value, "sample"))
        try:
            record = cls(
                sample_id=str(data["sample_id"]),
                embryo_id=str(data["embryo_id"]),
                field_of_view_id=str(data["field_of_view_id"]),
                image_relpath=str(data["image_relpath"]),
                truth_relpath=str(data["truth_relpath"]),
                shape_tzyx=tuple(data["shape_tzyx"]),
                dtype=str(data["dtype"]),
                chunks_tzyx=tuple(data["chunks_tzyx"]),
                scale_zyx_um=tuple(str(item) for item in data["scale_zyx_um"]),
                estimated_number_of_nodes=str(data["estimated_number_of_nodes"]),
                gt_node_count=int(data["gt_node_count"]),
                gt_edge_count=int(data["gt_edge_count"]),
                image_metadata_sha256=str(data["image_metadata_sha256"]),
                geff_tree_sha256=str(data["geff_tree_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError("MANIFEST_SCHEMA_INVALID", "sample fields") from exc
        _validate_sample_record(record)
        return record


@dataclass(frozen=True)
class FoldRecord:
    fold_id: str
    train_embryo_id: str
    evaluation_embryo_id: str
    train_membership: tuple[str, ...]
    calibration_membership: tuple[str, ...]
    threshold_selection_membership: tuple[str, ...]
    early_stopping_membership: tuple[str, ...]
    evaluation_membership: tuple[str, ...]
    train_membership_sha256: str
    calibration_membership_sha256: str
    evaluation_membership_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "train_membership",
            "calibration_membership",
            "threshold_selection_membership",
            "early_stopping_membership",
            "evaluation_membership",
        ):
            value[name] = list(value[name])
        return value

    @classmethod
    def from_dict(cls, value: Any) -> "FoldRecord":
        data = dict(_mapping(value, "fold"))
        try:
            return cls(
                fold_id=str(data["fold_id"]),
                train_embryo_id=str(data["train_embryo_id"]),
                evaluation_embryo_id=str(data["evaluation_embryo_id"]),
                train_membership=tuple(str(item) for item in data["train_membership"]),
                calibration_membership=tuple(str(item) for item in data["calibration_membership"]),
                threshold_selection_membership=tuple(
                    str(item) for item in data["threshold_selection_membership"]
                ),
                early_stopping_membership=tuple(str(item) for item in data["early_stopping_membership"]),
                evaluation_membership=tuple(str(item) for item in data["evaluation_membership"]),
                train_membership_sha256=str(data["train_membership_sha256"]),
                calibration_membership_sha256=str(data["calibration_membership_sha256"]),
                evaluation_membership_sha256=str(data["evaluation_membership_sha256"]),
            )
        except (KeyError, TypeError) as exc:
            raise ManifestError("MANIFEST_SCHEMA_INVALID", "fold fields") from exc


@dataclass(frozen=True)
class EvaluationManifest:
    scorer_lock_sha256: str
    source_inventory_sha256: str
    samples: tuple[SampleRecord, ...]
    folds: tuple[FoldRecord, ...]
    overlap_audit: Mapping[str, Any]
    manifest_sha256: str
    created_at: str
    schema_version: int = SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scorer_lock_sha256": self.scorer_lock_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "samples": [sample.to_dict() for sample in self.samples],
            "folds": [fold.to_dict() for fold in self.folds],
            "overlap_audit": dict(self.overlap_audit),
        }

    def computed_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.semantic_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_dict(),
            "manifest_sha256": self.manifest_sha256,
            "evidence": {"created_at": self.created_at},
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationManifest":
        data = dict(_mapping(value, "manifest"))
        evidence = _mapping(data.get("evidence"), "manifest.evidence")
        try:
            manifest = cls(
                schema_version=int(data["schema_version"]),
                scorer_lock_sha256=str(data["scorer_lock_sha256"]),
                source_inventory_sha256=str(data["source_inventory_sha256"]),
                samples=tuple(SampleRecord.from_dict(item) for item in data["samples"]),
                folds=tuple(FoldRecord.from_dict(item) for item in data["folds"]),
                overlap_audit=dict(_mapping(data["overlap_audit"], "overlap_audit")),
                manifest_sha256=str(data["manifest_sha256"]),
                created_at=str(evidence["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError("MANIFEST_SCHEMA_INVALID", "required fields") from exc
        _validate_manifest(manifest)
        return manifest


def _validate_sample_record(record: SampleRecord) -> None:
    match = _SAMPLE_ID.fullmatch(record.sample_id)
    if not match or match.group(1) != record.embryo_id or match.group(2) != record.field_of_view_id:
        _fail("IDENTITY_CONFLICT", record.sample_id)
    if Path(record.image_relpath).is_absolute() or Path(record.truth_relpath).is_absolute():
        _fail("PATH_ESCAPE", record.sample_id)
    if ".." in Path(record.image_relpath).parts or ".." in Path(record.truth_relpath).parts:
        _fail("PATH_ESCAPE", record.sample_id)
    if len(record.shape_tzyx) != 4 or any(item <= 0 for item in record.shape_tzyx):
        _fail("INVALID_METADATA", f"{record.sample_id}: shape")
    if len(record.chunks_tzyx) != 4 or any(item <= 0 for item in record.chunks_tzyx):
        _fail("INVALID_METADATA", f"{record.sample_id}: chunks")
    if record.gt_node_count < 0 or record.gt_edge_count < 0:
        _fail("INVALID_METADATA", f"{record.sample_id}: counts")
    for digest in (record.image_metadata_sha256, record.geff_tree_sha256):
        if not _SHA256.fullmatch(digest):
            _fail("MANIFEST_SCHEMA_INVALID", f"{record.sample_id}: digest")


def _membership_sha(samples: Mapping[str, SampleRecord], members: Sequence[str]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "sample_id": sample_id,
                    "image_metadata_sha256": samples[sample_id].image_metadata_sha256,
                    "geff_tree_sha256": samples[sample_id].geff_tree_sha256,
                }
                for sample_id in sorted(members)
            ]
        )
    )


def _validate_folds(samples: Mapping[str, SampleRecord], folds: Sequence[FoldRecord]) -> None:
    expected_ids = ("fold-44b6-to-6bba", "fold-6bba-to-44b6")
    if tuple(fold.fold_id for fold in folds) != expected_ids:
        _fail("RECIPROCAL_FOLDS_INVALID", "only the two official reciprocal folds are supported")
    evaluation_union: list[str] = []
    for fold in folds:
        train = tuple(sorted(fold.train_membership))
        evaluation = tuple(sorted(fold.evaluation_membership))
        if train != fold.train_membership or evaluation != fold.evaluation_membership:
            _fail("MEMBERSHIP_NOT_CANONICAL", fold.fold_id)
        if not train or not evaluation or set(train) & set(evaluation):
            _fail("SOURCE_OVERLAP", fold.fold_id)
        if any(samples[item].embryo_id != fold.train_embryo_id for item in train):
            _fail("IDENTITY_CONFLICT", f"{fold.fold_id}: train")
        if any(samples[item].embryo_id != fold.evaluation_embryo_id for item in evaluation):
            _fail("IDENTITY_CONFLICT", f"{fold.fold_id}: evaluation")
        for name, members in (
            ("calibration", fold.calibration_membership),
            ("threshold", fold.threshold_selection_membership),
            ("early_stopping", fold.early_stopping_membership),
        ):
            if tuple(sorted(members)) != members or not set(members) <= set(train):
                _fail("CALIBRATION_LEAKAGE", f"{fold.fold_id}: {name}")
        expected_hashes = (
            _membership_sha(samples, train),
            _membership_sha(samples, fold.calibration_membership),
            _membership_sha(samples, evaluation),
        )
        actual_hashes = (
            fold.train_membership_sha256,
            fold.calibration_membership_sha256,
            fold.evaluation_membership_sha256,
        )
        if actual_hashes != expected_hashes:
            _fail("MEMBERSHIP_HASH_MISMATCH", fold.fold_id)
        evaluation_union.extend(evaluation)
    if sorted(evaluation_union) != sorted(samples) or len(evaluation_union) != len(samples):
        _fail("RECIPROCAL_COVERAGE_INVALID", "each movie must appear once across evaluation sides")


def _validate_manifest(manifest: EvaluationManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        _fail("MANIFEST_SCHEMA_INVALID", "unsupported schema version")
    if not _SHA256.fullmatch(manifest.scorer_lock_sha256):
        _fail("MANIFEST_SCHEMA_INVALID", "scorer lock digest")
    if not _SHA256.fullmatch(manifest.source_inventory_sha256):
        _fail("MANIFEST_SCHEMA_INVALID", "source inventory digest")
    sample_ids = tuple(sample.sample_id for sample in manifest.samples)
    if not sample_ids or sample_ids != tuple(sorted(sample_ids)) or len(set(sample_ids)) != len(sample_ids):
        _fail("DUPLICATE_IDENTITY", "sample IDs must be unique and sorted")
    samples = {sample.sample_id: sample for sample in manifest.samples}
    paths = [item for sample in manifest.samples for item in (sample.image_relpath, sample.truth_relpath)]
    if len(set(paths)) != len(paths):
        _fail("DUPLICATE_PATH", "artifact path reused")
    geff_hashes = [sample.geff_tree_sha256 for sample in manifest.samples]
    if len(set(geff_hashes)) != len(geff_hashes):
        _fail("DUPLICATE_ARTIFACT_HASH", "GEFF content reused across sample identities")
    _validate_folds(samples, manifest.folds)
    if manifest.overlap_audit != {
        "cross_side_artifact_hashes": [],
        "cross_side_paths": [],
        "duplicate_sample_ids": [],
        "evaluation_union_complete": True,
        "movie_splitting": False,
        "passed": True,
    }:
        _fail("OVERLAP_AUDIT_INVALID", "audit is incomplete or reports overlap")
    if not secrets.compare_digest(manifest.manifest_sha256, manifest.computed_sha256()):
        _fail("MANIFEST_HASH_MISMATCH", "semantic manifest changed")


def _read_zarr_metadata(path: Path, expected_scale: tuple[str, str, str]) -> dict[str, Any]:
    root = _json(path / "zarr.json")
    attrs = _mapping(root.get("attributes"), f"{path}.attributes")
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or len(multiscales) != 1:
        _fail("INVALID_METADATA", f"{path}: multiscales")
    scale: Any = None
    axes = multiscales[0].get("axes") if isinstance(multiscales[0], Mapping) else None
    datasets = multiscales[0].get("datasets") if isinstance(multiscales[0], Mapping) else None
    axis_names = (
        [str(item.get("name", "")).casefold() for item in axes]
        if isinstance(axes, list) and all(isinstance(item, Mapping) for item in axes)
        else []
    )
    if axis_names != list(OFFICIAL_AXES):
        _fail("INVALID_METADATA", f"{path}: axes")
    if isinstance(datasets, list) and len(datasets) == 1 and isinstance(datasets[0], Mapping):
        transforms = datasets[0].get("coordinateTransformations")
        if isinstance(transforms, list) and transforms and isinstance(transforms[0], Mapping):
            scale = transforms[0].get("scale")
    if not isinstance(scale, list) or len(scale) != 4:
        _fail("INVALID_METADATA", f"{path}: scale")
    normalized_scale = tuple(_decimal_text(item) for item in scale[-3:])
    if normalized_scale != expected_scale:
        _fail("INVALID_METADATA", f"{path}: official scale changed")
    array = _json(path / "0" / "zarr.json")
    shape = _positive_ints(array.get("shape"), f"{path}: shape", 4)
    chunk_grid = _mapping(array.get("chunk_grid"), f"{path}: chunk_grid")
    chunk_config = _mapping(chunk_grid.get("configuration"), f"{path}: chunk configuration")
    chunks = _positive_ints(chunk_config.get("chunk_shape"), f"{path}: chunks", 4)
    dtype = str(array.get("data_type", ""))
    if dtype not in {"uint16", "<u2", "u2"}:
        _fail("INVALID_METADATA", f"{path}: dtype")
    return {"axes_tzyx": list(OFFICIAL_AXES), "chunks_tzyx": list(chunks), "dtype": "uint16", "scale_zyx_um": list(normalized_scale), "shape_tzyx": list(shape)}


def _read_geff_metadata(path: Path) -> dict[str, Any]:
    root = _json(path / "zarr.json")
    attrs = _mapping(root.get("attributes"), f"{path}.attributes")
    geff = _mapping(attrs.get("geff"), f"{path}.geff")
    if geff.get("directed") is not True:
        _fail("INVALID_METADATA", f"{path}: graph must be directed")
    axes = geff.get("axes")
    if not isinstance(axes, list) or [item.get("name") for item in axes if isinstance(item, Mapping)] != list(OFFICIAL_AXES):
        _fail("INVALID_METADATA", f"{path}: GEFF axes")
    extra = _mapping(geff.get("extra"), f"{path}.extra")
    estimate = _finite_positive(extra.get("estimated_number_of_nodes"), "estimated_number_of_nodes")
    props = _mapping(geff.get("node_props_metadata"), f"{path}.node_props_metadata")
    if not set(OFFICIAL_AXES) <= set(props):
        _fail("INVALID_METADATA", f"{path}: node properties")
    for relative in ("nodes/ids", "edges/ids", "nodes/props/t", "nodes/props/z", "nodes/props/y", "nodes/props/x"):
        if not (path / relative / "zarr.json").is_file():
            _fail("INVALID_METADATA", f"{path}: missing {relative}")
    nodes_shape = _array_shape(path, "nodes/ids", dimensions=1)
    edges_shape = _array_shape(path, "edges/ids", dimensions=2)
    if edges_shape[1] != 2:
        _fail("INVALID_METADATA", f"{path}: edges shape")
    claimed_sample_id = extra.get("sample_id")
    if claimed_sample_id is not None and not isinstance(claimed_sample_id, str):
        _fail("INVALID_METADATA", f"{path}: extra.sample_id")
    return {
        "estimated_number_of_nodes": _decimal_text(estimate),
        "gt_node_count": nodes_shape[0],
        "gt_edge_count": edges_shape[0],
        "claimed_sample_id": claimed_sample_id,
    }


def _discover_samples(data_root: Path, scorer_lock: Mapping[str, Any]) -> tuple[SampleRecord, ...]:
    root = data_root.resolve(strict=True)
    constants = _mapping(scorer_lock.get("constants"), "scorer lock constants")
    raw_scale = constants.get("scale_zyx_um")
    if not isinstance(raw_scale, list) or len(raw_scale) != 3:
        _fail("SCORER_LOCK_INVALID", "scale_zyx_um")
    expected_scale = tuple(str(item) for item in raw_scale)
    zarr_paths = sorted(root.glob("*.zarr"), key=lambda path: path.name)
    geff_paths = sorted(root.glob("*.geff"), key=lambda path: path.name)
    zarr = {path.stem: path for path in zarr_paths}
    geff = {path.stem: path for path in geff_paths}
    if set(zarr) != set(geff):
        _fail("UNPAIRED_ARTIFACT", f"zarr-only={sorted(set(zarr)-set(geff))}, geff-only={sorted(set(geff)-set(zarr))}")
    if not zarr:
        _fail("NO_SAMPLES", str(root))
    records: list[SampleRecord] = []
    for sample_id in sorted(zarr):
        match = _SAMPLE_ID.fullmatch(sample_id)
        if not match:
            _fail("UNKNOWN_IDENTITY", sample_id)
        image_path = _contained(zarr[sample_id], root)
        truth_path = _contained(geff[sample_id], root)
        image_metadata = _read_zarr_metadata(image_path, expected_scale)
        truth_metadata = _read_geff_metadata(truth_path)
        if truth_metadata["claimed_sample_id"] not in {None, sample_id}:
            _fail("IDENTITY_CONFLICT", f"{sample_id} != {truth_metadata['claimed_sample_id']}")
        record = SampleRecord(
            sample_id=sample_id,
            embryo_id=match.group(1),
            field_of_view_id=match.group(2),
            image_relpath=image_path.relative_to(root).as_posix(),
            truth_relpath=truth_path.relative_to(root).as_posix(),
            shape_tzyx=tuple(image_metadata["shape_tzyx"]),
            dtype=image_metadata["dtype"],
            chunks_tzyx=tuple(image_metadata["chunks_tzyx"]),
            scale_zyx_um=tuple(image_metadata["scale_zyx_um"]),
            estimated_number_of_nodes=truth_metadata["estimated_number_of_nodes"],
            gt_node_count=truth_metadata["gt_node_count"],
            gt_edge_count=truth_metadata["gt_edge_count"],
            image_metadata_sha256=sha256_bytes(canonical_json_bytes(image_metadata)),
            geff_tree_sha256=_tree_sha256(truth_path),
        )
        _validate_sample_record(record)
        records.append(record)
    return tuple(records)


def _folds(
    samples: tuple[SampleRecord, ...],
    calibration_by_fold: Mapping[str, Sequence[str]] | None,
) -> tuple[FoldRecord, ...]:
    sample_map = {sample.sample_id: sample for sample in samples}
    result: list[FoldRecord] = []
    for train_embryo, evaluation_embryo in (("44b6", "6bba"), ("6bba", "44b6")):
        fold_id = f"fold-{train_embryo}-to-{evaluation_embryo}"
        train = tuple(sorted(item.sample_id for item in samples if item.embryo_id == train_embryo))
        evaluation = tuple(sorted(item.sample_id for item in samples if item.embryo_id == evaluation_embryo))
        calibration = tuple(
            sorted((calibration_by_fold or {}).get(fold_id, train))
        )
        if not set(calibration) <= set(train):
            _fail("CALIBRATION_LEAKAGE", fold_id)
        result.append(
            FoldRecord(
                fold_id=fold_id,
                train_embryo_id=train_embryo,
                evaluation_embryo_id=evaluation_embryo,
                train_membership=train,
                calibration_membership=calibration,
                threshold_selection_membership=calibration,
                early_stopping_membership=calibration,
                evaluation_membership=evaluation,
                train_membership_sha256=_membership_sha(sample_map, train),
                calibration_membership_sha256=_membership_sha(sample_map, calibration),
                evaluation_membership_sha256=_membership_sha(sample_map, evaluation),
            )
        )
    return tuple(result)


def build_manifest(
    data_root: str | Path,
    scorer_lock_path: str | Path,
    *,
    calibration_by_fold: Mapping[str, Sequence[str]] | None = None,
    created_at: datetime | None = None,
) -> EvaluationManifest:
    lock_path = Path(scorer_lock_path).resolve(strict=True)
    scorer_lock = _json(lock_path, "SCORER_LOCK_INVALID")
    samples = _discover_samples(Path(data_root), scorer_lock)
    folds = _folds(samples, calibration_by_fold)
    semantic_samples = [sample.to_dict() for sample in samples]
    source_inventory_sha256 = sha256_bytes(canonical_json_bytes(semantic_samples))
    manifest = EvaluationManifest(
        scorer_lock_sha256=sha256_bytes(canonical_json_bytes(scorer_lock)),
        source_inventory_sha256=source_inventory_sha256,
        samples=samples,
        folds=folds,
        overlap_audit={
            "cross_side_artifact_hashes": [],
            "cross_side_paths": [],
            "duplicate_sample_ids": [],
            "evaluation_union_complete": True,
            "movie_splitting": False,
            "passed": True,
        },
        manifest_sha256="",
        created_at=(created_at or utc_now()).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    manifest = EvaluationManifest(**{**manifest.__dict__, "manifest_sha256": manifest.computed_sha256()})
    _validate_manifest(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: EvaluationManifest) -> Path:
    _validate_manifest(manifest)
    return atomic_write_json(path, manifest.to_dict())


def load_manifest(path: str | Path) -> EvaluationManifest:
    return EvaluationManifest.from_dict(_json(Path(path), "MANIFEST_UNREADABLE"))


def verify_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    scorer_lock_path: str | Path | None = None,
) -> EvaluationManifest:
    manifest = load_manifest(manifest_path)
    if data_root is not None:
        if scorer_lock_path is None:
            _fail("SCORER_LOCK_REQUIRED", "data verification requires the scorer lock")
        rebuilt = build_manifest(
            data_root,
            scorer_lock_path,
            calibration_by_fold={fold.fold_id: fold.calibration_membership for fold in manifest.folds},
        )
        if rebuilt.semantic_dict() != manifest.semantic_dict():
            _fail("SOURCE_INVENTORY_MISMATCH", "mounted data no longer matches manifest")
    return manifest
