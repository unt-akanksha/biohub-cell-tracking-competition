from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import stat
import struct
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file, utc_now
from .ledger import (
    CpuAcceptanceStatus,
    EventType,
    ExactEvaluationStatus,
    ExperimentEvent,
    Ledger,
    cpu_acceptance_completed_payload,
    cpu_acceptance_failed_payload,
    cpu_acceptance_inputs_bound_payload,
    cpu_acceptance_registration_payload,
    event_sha256,
    exact_evaluation_completed_payload,
    exact_evaluation_failed_payload,
    exact_evaluation_registration_payload,
    exact_evaluation_started_payload,
    reconstruct_cpu_acceptances,
    reconstruct_exact_evaluations,
    resolved_exact_member,
)


PENDING_SCHEMA = "biohub.pending-control-report.v1"
PENDING_ENVELOPE_SCHEMA = "biohub.pending-control-envelope.v1"
BUNDLE_SCHEMA = "biohub.runtime-bundle.v1"
BUNDLE_MAGIC = b"BIOHUB-RUNTIME-BUNDLE\x00\x01"
BUNDLE_NAME = "biohub-runtime-v1.biohubbundle"
BUNDLE_MAX_UNCOMPRESSED_BYTES = 2_147_483_648
BUNDLE_ALLOWED_ROOTS = frozenset(
    {"config", "kaggle", "requirements", "site-packages", "src", "tests", "vendor"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PENDING_KEYS = {
    "schema_version",
    "status",
    "run_id",
    "evaluation_run_id",
    "request_nonce",
    "acceptance_request_sha256",
    "registration_event_sha256",
    "kernel_ref",
    "runtime_dataset_ref",
    "runtime_bundle_name",
    "runtime_bundle_sha256",
    "runtime_bundle_inventory_sha256",
    "runtime_bundle_uncompressed_size_bytes",
    "runtime_bundle_file_count",
    "accelerator",
    "internet_enabled",
    "competition_submission_performed",
    "watchdog_terminal_state",
    "actual_cpu_runtime_seconds",
    "peak_memory_mb",
    "source_identities",
    "manifest",
    "control",
    "pending_payload_sha256",
    "pending_envelope",
    "pending_envelope_sha256",
    "output_inventory_sha256",
}
_SOURCE_KEYS = {
    "scorer_lock_sha256",
    "environment_lock_sha256",
    "manifest_policy_sha256",
    "control_model_sha256",
    "config_sha256",
    "code_sha256",
    "data_source_sha256",
}
_FORBIDDEN_SOURCE = (
    "kaggle competitions submit",
    "biohub launch execute",
    "torch.cuda",
    ".cuda(",
    "cuda.is_available",
)


class AcceptanceError(ValueError):
    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def _fail(reason_code: str, detail: str) -> None:
    raise AcceptanceError(reason_code, detail)


def _sha(value: Any, name: str) -> str:
    digest = str(value).casefold()
    if _SHA256_RE.fullmatch(digest) is None:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", name)
    return digest


def _text(value: Any, name: str, limit: int = 240) -> str:
    result = str(value).strip()
    if not result or len(result) > limit:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", name)
    return result


def _positive_decimal(value: Any, name: str) -> str:
    if isinstance(value, bool):
        _fail("PENDING_CONTROL_SCHEMA_INVALID", name)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AcceptanceError("PENDING_CONTROL_SCHEMA_INVALID", name) from exc
    if not result.is_finite() or result <= 0:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", name)
    return format(result, "f")


def _finite_tree(value: Any, path: str = "pending") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail("PENDING_CONTROL_SCHEMA_INVALID", path)


@dataclass(frozen=True)
class PendingControlReport:
    value: Mapping[str, Any]
    run_id: str
    evaluation_run_id: str
    request_nonce: str
    acceptance_request_sha256: str
    registration_event_sha256: str
    kernel_ref: str
    runtime_dataset_ref: str
    runtime_bundle_name: str
    runtime_bundle_sha256: str
    runtime_bundle_inventory_sha256: str
    runtime_bundle_uncompressed_size_bytes: int
    runtime_bundle_file_count: int
    accelerator: str
    pending_payload_sha256: str
    pending_envelope_sha256: str
    output_inventory_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.value)


def validate_pending_control(value: Any) -> PendingControlReport:
    if not isinstance(value, dict) or set(value) != _PENDING_KEYS:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "unknown or missing root field")
    if value["schema_version"] != PENDING_SCHEMA:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "schema_version")
    if value["status"] != "pending_reconciliation":
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "status")
    if value["accelerator"] != "none":
        _fail("ACCELERATOR_FORBIDDEN", str(value["accelerator"]))
    if value["internet_enabled"] is not False:
        _fail("INTERNET_FORBIDDEN", "remote job enabled internet")
    if value["competition_submission_performed"] is not False:
        _fail("SUBMISSION_FORBIDDEN", "remote job reported a submission")
    if value["watchdog_terminal_state"] != "completed":
        _fail("REMOTE_JOB_NONTERMINAL", str(value["watchdog_terminal_state"]))
    _positive_decimal(value["actual_cpu_runtime_seconds"], "actual_cpu_runtime_seconds")
    _positive_decimal(value["peak_memory_mb"], "peak_memory_mb")
    if value["runtime_bundle_name"] != BUNDLE_NAME:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "runtime_bundle_name")
    bundle_size = value["runtime_bundle_uncompressed_size_bytes"]
    bundle_count = value["runtime_bundle_file_count"]
    if (
        not isinstance(bundle_size, int)
        or isinstance(bundle_size, bool)
        or bundle_size < 1
        or bundle_size > BUNDLE_MAX_UNCOMPRESSED_BYTES
        or not isinstance(bundle_count, int)
        or isinstance(bundle_count, bool)
        or bundle_count < 1
    ):
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "runtime bundle size/count")
    identities = value["source_identities"]
    if not isinstance(identities, dict) or set(identities) != _SOURCE_KEYS:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "source_identities")
    for name in sorted(_SOURCE_KEYS):
        _sha(identities[name], f"source_identities.{name}")
    manifest = value["manifest"]
    if not isinstance(manifest, dict) or set(manifest) != {
        "manifest_sha256",
        "folds",
        "sample_count",
        "overlap_count",
        "manifest_document",
    }:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "manifest")
    _sha(manifest["manifest_sha256"], "manifest.manifest_sha256")
    if not isinstance(manifest["sample_count"], int) or manifest["sample_count"] <= 0:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "manifest.sample_count")
    if manifest["overlap_count"] != 0:
        _fail("MANIFEST_OVERLAP", str(manifest["overlap_count"]))
    folds = manifest["folds"]
    fold_keys = {
        "fold_id",
        "train_membership_sha256",
        "calibration_membership_sha256",
        "evaluation_membership_sha256",
    }
    if not isinstance(folds, list) or len(folds) != 2:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "manifest.folds")
    if folds != sorted(folds, key=lambda item: str(item.get("fold_id", ""))):
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "manifest fold order")
    for fold in folds:
        if not isinstance(fold, dict) or set(fold) != fold_keys:
            _fail("PENDING_CONTROL_SCHEMA_INVALID", "manifest fold")
        _text(fold["fold_id"], "fold_id")
        for name in fold_keys - {"fold_id"}:
            _sha(fold[name], name)
    if not isinstance(value["control"], dict):
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "control")
    _finite_tree(value)

    semantic = dict(value)
    for name in (
        "pending_payload_sha256",
        "pending_envelope",
        "pending_envelope_sha256",
        "output_inventory_sha256",
    ):
        semantic.pop(name)
    payload_sha = _sha(value["pending_payload_sha256"], "pending_payload_sha256")
    if not secrets.compare_digest(payload_sha, sha256_bytes(canonical_json_bytes(semantic))):
        _fail("PENDING_PAYLOAD_HASH_MISMATCH", "semantic pending payload")
    envelope = value["pending_envelope"]
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version",
        "pending_payload_sha256",
        "authority",
    }:
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "pending_envelope")
    if (
        envelope["schema_version"] != PENDING_ENVELOPE_SCHEMA
        or envelope["authority"] != "remote_untrusted_pending"
        or envelope["pending_payload_sha256"] != payload_sha
    ):
        _fail("PENDING_CONTROL_SCHEMA_INVALID", "pending_envelope authority")
    envelope_sha = _sha(value["pending_envelope_sha256"], "pending_envelope_sha256")
    if not secrets.compare_digest(
        envelope_sha, sha256_bytes(canonical_json_bytes(envelope))
    ):
        _fail("PENDING_ENVELOPE_HASH_MISMATCH", "pending envelope")
    output_semantic = dict(value)
    output_semantic.pop("output_inventory_sha256")
    output_sha = _sha(value["output_inventory_sha256"], "output_inventory_sha256")
    if not secrets.compare_digest(
        output_sha, sha256_bytes(canonical_json_bytes(output_semantic))
    ):
        _fail("OUTPUT_INVENTORY_HASH_MISMATCH", "compact output")
    return PendingControlReport(
        value=value,
        run_id=_text(value["run_id"], "run_id", 160),
        evaluation_run_id=_text(value["evaluation_run_id"], "evaluation_run_id", 160),
        request_nonce=_sha(value["request_nonce"], "request_nonce"),
        acceptance_request_sha256=_sha(
            value["acceptance_request_sha256"], "acceptance_request_sha256"
        ),
        registration_event_sha256=_sha(
            value["registration_event_sha256"], "registration_event_sha256"
        ),
        kernel_ref=_text(value["kernel_ref"], "kernel_ref"),
        runtime_dataset_ref=_text(value["runtime_dataset_ref"], "runtime_dataset_ref"),
        runtime_bundle_name=BUNDLE_NAME,
        runtime_bundle_sha256=_sha(
            value["runtime_bundle_sha256"], "runtime_bundle_sha256"
        ),
        runtime_bundle_inventory_sha256=_sha(
            value["runtime_bundle_inventory_sha256"],
            "runtime_bundle_inventory_sha256",
        ),
        runtime_bundle_uncompressed_size_bytes=bundle_size,
        runtime_bundle_file_count=bundle_count,
        accelerator="none",
        pending_payload_sha256=payload_sha,
        pending_envelope_sha256=envelope_sha,
        output_inventory_sha256=output_sha,
    )


def load_pending_control(path: str | Path) -> PendingControlReport:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError("PENDING_CONTROL_UNREADABLE", str(path)) from exc
    return validate_pending_control(value)


def assert_cpu_kernel_metadata(
    metadata: Any,
    *,
    expected_kernel_slug: str,
    expected_dataset_slug: str,
    expected_competition_slug: str,
) -> None:
    if not isinstance(metadata, dict):
        _fail("KERNEL_METADATA_INVALID", "root")
    if metadata.get("id") != expected_kernel_slug:
        _fail("KERNEL_OWNERSHIP_MISMATCH", str(metadata.get("id")))
    for field in ("enable_gpu", "enable_tpu", "enable_internet"):
        if metadata.get(field) is not False:
            _fail("KERNEL_METADATA_INVALID", f"{field} must be false")
    if metadata.get("kernel_type") != "script" or metadata.get("language") != "python":
        _fail("KERNEL_METADATA_INVALID", "CPU acceptance must be a Python script")
    if metadata.get("is_private") is not True:
        _fail("KERNEL_METADATA_INVALID", "acceptance kernel must be private")
    if metadata.get("dataset_sources") != [expected_dataset_slug]:
        _fail("KERNEL_METADATA_INVALID", "runtime dataset source")
    if metadata.get("competition_sources") != [expected_competition_slug]:
        _fail("KERNEL_METADATA_INVALID", "competition source")


def assert_no_submission_source(source: str) -> None:
    lowered = source.casefold()
    for token in _FORBIDDEN_SOURCE:
        if token in lowered:
            _fail("FORBIDDEN_EXECUTION_PATH", token)


def _semantic_file_sha(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return sha256_file(path)
    return sha256_bytes(canonical_json_bytes(value))


def _bundle_member_path(value: str) -> PurePosixPath:
    if "\\" in value:
        _fail("RUNTIME_BUNDLE_PATH_INVALID", value)
    path = PurePosixPath(value)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or parts[0] != "bundle"
        or len(parts) < 3
        or parts[1] not in BUNDLE_ALLOWED_ROOTS
        or any(part in {"", ".", ".."} for part in parts)
    ):
        _fail("RUNTIME_BUNDLE_PATH_INVALID", value)
    return path


def _runtime_bundle_files(source_root: Path) -> list[tuple[str, Path, int, str]]:
    source = source_root.resolve(strict=True)
    bundle_root = source / "bundle"
    if not bundle_root.is_dir():
        _fail("RUNTIME_BUNDLE_SOURCE_INVALID", "missing bundle directory")
    records: list[tuple[str, Path, int, str]] = []
    seen: set[str] = set()
    for path in sorted(bundle_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(source).as_posix()
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise AcceptanceError("RUNTIME_BUNDLE_SOURCE_UNREADABLE", relative) from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if path.is_symlink() or (reparse and attributes & reparse):
            _fail("RUNTIME_BUNDLE_LINK_FORBIDDEN", relative)
        if path.is_dir():
            parts = PurePosixPath(relative).parts
            if (
                len(parts) < 2
                or parts[0] != "bundle"
                or parts[1] not in BUNDLE_ALLOWED_ROOTS
            ):
                _fail("RUNTIME_BUNDLE_PATH_INVALID", relative)
            continue
        _bundle_member_path(relative)
        if not stat.S_ISREG(metadata.st_mode):
            _fail("RUNTIME_BUNDLE_FILE_TYPE_FORBIDDEN", relative)
        if int(getattr(metadata, "st_nlink", 1)) != 1:
            _fail("RUNTIME_BUNDLE_HARDLINK_FORBIDDEN", relative)
        folded = relative.casefold()
        if folded in seen:
            _fail("RUNTIME_BUNDLE_DUPLICATE_PATH", relative)
        seen.add(folded)
        records.append((relative, path, metadata.st_size, sha256_file(path)))
    if not records:
        _fail("RUNTIME_BUNDLE_SOURCE_INVALID", "empty bundle")
    size = sum(item[2] for item in records)
    if size > BUNDLE_MAX_UNCOMPRESSED_BYTES:
        _fail("RUNTIME_BUNDLE_EXPANSION_LIMIT", str(size))
    return records


def _bundle_inventory(records: list[tuple[str, Path, int, str]]) -> dict[str, Any]:
    values = [
        {"path": relative, "size_bytes": size, "sha256": digest}
        for relative, _, size, digest in records
    ]
    return {
        "file_count": len(values),
        "uncompressed_size_bytes": sum(item["size_bytes"] for item in values),
        "inventory_sha256": sha256_bytes(canonical_json_bytes(values)),
    }


def _read_bundle_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb") as handle:
            magic = handle.read(len(BUNDLE_MAGIC))
            if magic != BUNDLE_MAGIC:
                _fail("RUNTIME_BUNDLE_MAGIC_INVALID", path.name)
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                _fail("RUNTIME_BUNDLE_HEADER_INVALID", "length")
            header_length = struct.unpack(">Q", raw_length)[0]
            if header_length < 2 or header_length > 65_536:
                _fail("RUNTIME_BUNDLE_HEADER_INVALID", "length")
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                _fail("RUNTIME_BUNDLE_HEADER_INVALID", "truncated")
            header = json.loads(raw_header.decode("utf-8"))
            payload_offset = handle.tell()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError("RUNTIME_BUNDLE_HEADER_INVALID", str(path)) from exc
    required = {
        "schema_version",
        "compression",
        "file_count",
        "uncompressed_size_bytes",
        "inventory_sha256",
        "payload_size_bytes",
        "payload_sha256",
        "tar_size_bytes",
    }
    if not isinstance(header, dict) or set(header) != required:
        _fail("RUNTIME_BUNDLE_HEADER_INVALID", "fields")
    if header["schema_version"] != BUNDLE_SCHEMA or header["compression"] != "zlib":
        _fail("RUNTIME_BUNDLE_HEADER_INVALID", "schema or compression")
    for name in ("file_count", "uncompressed_size_bytes", "payload_size_bytes", "tar_size_bytes"):
        if not isinstance(header[name], int) or header[name] <= 0:
            _fail("RUNTIME_BUNDLE_HEADER_INVALID", name)
    if header["uncompressed_size_bytes"] > BUNDLE_MAX_UNCOMPRESSED_BYTES:
        _fail("RUNTIME_BUNDLE_EXPANSION_LIMIT", str(header["uncompressed_size_bytes"]))
    _sha(header["inventory_sha256"], "inventory_sha256")
    _sha(header["payload_sha256"], "payload_sha256")
    if path.stat().st_size - payload_offset != header["payload_size_bytes"]:
        _fail("RUNTIME_BUNDLE_HEADER_INVALID", "payload_size_bytes")
    return header, payload_offset


def inspect_runtime_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).resolve(strict=True)
    header, _ = _read_bundle_header(bundle_path)
    return {
        "runtime_bundle_name": bundle_path.name,
        "runtime_bundle_sha256": sha256_file(bundle_path),
        "runtime_bundle_inventory_sha256": header["inventory_sha256"],
        "runtime_bundle_uncompressed_size_bytes": header["uncompressed_size_bytes"],
        "runtime_bundle_file_count": header["file_count"],
    }


def build_runtime_bundle(
    source_root: str | Path, output_path: str | Path
) -> dict[str, Any]:
    source = Path(source_root).resolve(strict=True)
    output = Path(output_path).resolve()
    if output.name != BUNDLE_NAME:
        _fail("RUNTIME_BUNDLE_NAME_INVALID", output.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing existing runtime bundle: {output}")
    records = _runtime_bundle_files(source)
    inventory = _bundle_inventory(records)
    tar_path: Path | None = None
    payload_path: Path | None = None
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing existing partial runtime bundle: {partial}")
    try:
        with tempfile.NamedTemporaryFile(
            prefix="biohub-runtime-", suffix=".tar.tmp", dir=output.parent, delete=False
        ) as handle:
            tar_path = Path(handle.name)
        with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as archive:
            for relative, path, size, _ in records:
                info = tarfile.TarInfo(relative)
                info.size = size
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        with tempfile.NamedTemporaryFile(
            prefix="biohub-runtime-", suffix=".zlib.tmp", dir=output.parent, delete=False
        ) as handle:
            payload_path = Path(handle.name)
            compressor = zlib.compressobj(level=9)
            with tar_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    handle.write(compressor.compress(chunk))
            handle.write(compressor.flush())
        header = {
            "schema_version": BUNDLE_SCHEMA,
            "compression": "zlib",
            **inventory,
            "payload_size_bytes": payload_path.stat().st_size,
            "payload_sha256": sha256_file(payload_path),
            "tar_size_bytes": tar_path.stat().st_size,
        }
        raw_header = canonical_json_bytes(header)
        with partial.open("xb") as destination:
            destination.write(BUNDLE_MAGIC)
            destination.write(struct.pack(">Q", len(raw_header)))
            destination.write(raw_header)
            with payload_path.open("rb") as stream:
                shutil.copyfileobj(stream, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(partial, output)
    finally:
        for temporary in (tar_path, payload_path):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return inspect_runtime_bundle(output)


def _decompress_bundle_payload(bundle: Path, offset: int, header: Mapping[str, Any], tar_path: Path) -> None:
    digest = __import__("hashlib").sha256()
    decompressor = zlib.decompressobj()
    expanded = 0
    with bundle.open("rb") as source, tar_path.open("xb") as destination:
        source.seek(offset)
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            pending = chunk
            while pending:
                decoded = decompressor.decompress(pending, 1024 * 1024)
                pending = decompressor.unconsumed_tail
                expanded += len(decoded)
                if expanded > BUNDLE_MAX_UNCOMPRESSED_BYTES or expanded > header["tar_size_bytes"]:
                    _fail("RUNTIME_BUNDLE_EXPANSION_LIMIT", str(expanded))
                destination.write(decoded)
        tail = decompressor.flush()
        expanded += len(tail)
        destination.write(tail)
    if (
        not decompressor.eof
        or decompressor.unused_data
        or expanded != header["tar_size_bytes"]
        or digest.hexdigest() != header["payload_sha256"]
    ):
        _fail("RUNTIME_BUNDLE_PAYLOAD_INVALID", bundle.name)


def extract_runtime_bundle(
    bundle_path: str | Path,
    destination_root: str | Path,
    *,
    expected_sha256: str,
    expected_inventory_sha256: str,
    expected_uncompressed_size_bytes: int,
    expected_file_count: int,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve(strict=True)
    if bundle.name != BUNDLE_NAME:
        _fail("RUNTIME_BUNDLE_NAME_INVALID", bundle.name)
    if not secrets.compare_digest(sha256_file(bundle), _sha(expected_sha256, "runtime_bundle_sha256")):
        _fail("RUNTIME_BUNDLE_HASH_MISMATCH", bundle.name)
    header, offset = _read_bundle_header(bundle)
    expected = {
        "inventory_sha256": _sha(expected_inventory_sha256, "runtime_bundle_inventory_sha256"),
        "uncompressed_size_bytes": int(expected_uncompressed_size_bytes),
        "file_count": int(expected_file_count),
    }
    if any(header[name] != value for name, value in expected.items()):
        _fail("RUNTIME_BUNDLE_INVENTORY_MISMATCH", bundle.name)
    destination = Path(destination_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_bundle = destination / "bundle"
    if final_bundle.exists():
        raise FileExistsError(f"refusing existing extracted bundle: {final_bundle}")
    temporary = Path(tempfile.mkdtemp(prefix=".biohub-extract-", dir=destination))
    tar_path = temporary / "payload.tar"
    try:
        _decompress_bundle_payload(bundle, offset, header, tar_path)
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        total = 0
        with tarfile.open(tar_path, "r:") as archive:
            for member in archive:
                if not member.isfile():
                    _fail("RUNTIME_BUNDLE_FILE_TYPE_FORBIDDEN", member.name)
                relative = _bundle_member_path(member.name)
                folded = member.name.casefold()
                if folded in seen:
                    _fail("RUNTIME_BUNDLE_DUPLICATE_PATH", member.name)
                seen.add(folded)
                total += member.size
                if total > BUNDLE_MAX_UNCOMPRESSED_BYTES or total > expected["uncompressed_size_bytes"]:
                    _fail("RUNTIME_BUNDLE_EXPANSION_LIMIT", str(total))
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    _fail("RUNTIME_BUNDLE_FILE_TYPE_FORBIDDEN", member.name)
                hasher = __import__("hashlib").sha256()
                written = 0
                with target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > member.size:
                            _fail("RUNTIME_BUNDLE_MEMBER_SIZE_INVALID", member.name)
                        hasher.update(chunk)
                        output.write(chunk)
                if written != member.size:
                    _fail("RUNTIME_BUNDLE_MEMBER_SIZE_INVALID", member.name)
                records.append(
                    {"path": member.name, "size_bytes": written, "sha256": hasher.hexdigest()}
                )
        records.sort(key=lambda item: item["path"])
        actual = {
            "file_count": len(records),
            "uncompressed_size_bytes": total,
            "inventory_sha256": sha256_bytes(canonical_json_bytes(records)),
        }
        if actual != expected:
            _fail("RUNTIME_BUNDLE_INVENTORY_MISMATCH", bundle.name)
        tar_path.unlink()
        extracted_bundle = temporary / "bundle"
        try:
            os.replace(extracted_bundle, final_bundle)
        except PermissionError:
            # Windows can deny an otherwise same-volume directory rename after
            # scanning a large extracted tree. Every member has already passed
            # path/type/size/hash validation, so a non-link-following copy is a
            # safe portability fallback. The destination was required absent.
            shutil.copytree(extracted_bundle, final_bundle, symlinks=False)
        return {
            "runtime_bundle_name": bundle.name,
            "runtime_bundle_sha256": sha256_file(bundle),
            "runtime_bundle_inventory_sha256": actual["inventory_sha256"],
            "runtime_bundle_uncompressed_size_bytes": actual["uncompressed_size_bytes"],
            "runtime_bundle_file_count": actual["file_count"],
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _source_inventory_sha256(root: Path) -> str:
    included = [
        *sorted((root / "src" / "biohub_tracker").glob("*.py")),
        root / "kaggle" / "phase2-cpu-acceptance" / "phase2_acceptance.py",
        root / "config" / "official-scorer.lock.json",
        root / "config" / "evaluation-policy.json",
        root / "requirements" / "evaluation-lock.txt",
    ]
    records = []
    for path in included:
        path = path.resolve(strict=True)
        path.relative_to(root)
        records.append(
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        )
    return sha256_bytes(canonical_json_bytes(records))


def issue_acceptance_request(
    *,
    workspace_root: str | Path,
    ledger_path: str | Path,
    config_path: str | Path,
    run_id: str,
    evaluation_run_id: str,
    kernel_ref: str,
    runtime_dataset_ref: str,
    runtime_bundle_path: str | Path,
    output_path: str | Path,
) -> tuple[dict[str, Any], ExperimentEvent]:
    root = Path(workspace_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != "biohub.phase2-control-config.v1"
        or config.get("accelerator") != "none"
        or config.get("enable_gpu") is not False
        or config.get("enable_tpu") is not False
        or config.get("enable_internet") is not False
        or config.get("competition_submission_allowed") is not False
    ):
        _fail("CONTROL_CONFIG_UNSAFE", str(config_file))
    scorer_lock_path = root / str(config["scorer_lock_path"])
    environment_lock_path = root / str(config["environment_lock_path"])
    evaluation_policy_path = root / str(config["evaluation_policy_path"])
    scorer_lock = json.loads(scorer_lock_path.read_text(encoding="utf-8"))
    identities = {
        "scorer_lock_sha256": sha256_bytes(canonical_json_bytes(scorer_lock)),
        "environment_lock_sha256": sha256_file(environment_lock_path),
        "manifest_policy_sha256": sha256_bytes(
            canonical_json_bytes({"manifest_policy": config["manifest_policy"]})
        ),
        "control_model_sha256": sha256_bytes(
            canonical_json_bytes({"control_model": config["control_model"]})
        ),
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "code_sha256": _source_inventory_sha256(root),
        "data_source_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "competition_slug": config["competition_slug"],
                    "source": "mounted-official-train",
                }
            )
        ),
    }
    policy = json.loads(evaluation_policy_path.read_text(encoding="utf-8"))
    if sha256_bytes(canonical_json_bytes(policy)) == "0" * 64:  # pragma: no cover
        _fail("CONTROL_CONFIG_INVALID", "evaluation policy")
    bundle = inspect_runtime_bundle(runtime_bundle_path)
    if bundle["runtime_bundle_name"] != BUNDLE_NAME:
        _fail("RUNTIME_BUNDLE_NAME_INVALID", bundle["runtime_bundle_name"])
    nonce = secrets.token_hex(32)
    semantic = {
        "schema_version": "biohub.acceptance-request.v1",
        "run_id": _text(run_id, "run_id", 160),
        "evaluation_run_id": _text(evaluation_run_id, "evaluation_run_id", 160),
        "purpose": config["purpose"],
        "request_nonce": nonce,
        "competition_slug": config["competition_slug"],
        "kernel_slug": config["kernel_slug"],
        "runtime_dataset_slug": config["runtime_dataset_slug"],
        "kernel_ref": _text(kernel_ref, "kernel_ref"),
        "runtime_dataset_ref": _text(runtime_dataset_ref, "runtime_dataset_ref"),
        **bundle,
        "cpu_watchdog_minutes": int(config["cpu_watchdog_minutes"]),
        "accelerator": "none",
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_submission_allowed": False,
        "source_identities": identities,
    }
    request_sha = sha256_bytes(canonical_json_bytes(semantic))
    ledger = Ledger(Path(ledger_path), root)
    registration_payload = cpu_acceptance_registration_payload(
        run_id=semantic["run_id"],
        purpose=semantic["purpose"],
        request_nonce=nonce,
        acceptance_request_sha256=request_sha,
        evaluation_run_id=semantic["evaluation_run_id"],
        kernel_slug=semantic["kernel_slug"],
        runtime_dataset_slug=semantic["runtime_dataset_slug"],
        runtime_bundle_name=semantic["runtime_bundle_name"],
        runtime_bundle_sha256=semantic["runtime_bundle_sha256"],
        runtime_bundle_inventory_sha256=semantic[
            "runtime_bundle_inventory_sha256"
        ],
        runtime_bundle_uncompressed_size_bytes=semantic[
            "runtime_bundle_uncompressed_size_bytes"
        ],
        runtime_bundle_file_count=semantic["runtime_bundle_file_count"],
        scorer_lock_sha256=identities["scorer_lock_sha256"],
        environment_lock_sha256=identities["environment_lock_sha256"],
        manifest_policy_sha256=identities["manifest_policy_sha256"],
        control_model_sha256=identities["control_model_sha256"],
        config_sha256=identities["config_sha256"],
        code_sha256=identities["code_sha256"],
        data_source_sha256=identities["data_source_sha256"],
        cpu_watchdog_minutes=semantic["cpu_watchdog_minutes"],
    )
    event = ExperimentEvent.create(
        semantic["run_id"], EventType.CPU_ACCEPTANCE_REGISTERED, registration_payload
    )
    ledger.append(event)
    request = {
        **semantic,
        "acceptance_request_sha256": request_sha,
        "registration_event_sha256": event_sha256(event),
    }
    atomic_write_json(output_path, request)
    return request, event


def _rebind_official_producers(
    official: Mapping[str, Any], members: list[dict[str, Any]]
) -> dict[str, Any]:
    result = json.loads(json.dumps(official))
    by_slot = {(item["role"], item["fold_id"]): item for item in members}
    for role in ("baseline", "candidate"):
        for row in result[role]["by_movie"]:
            row["producer"] = by_slot[(role, row["fold_id"])]
    return result


def _gpu_quota_state(path: str | Path) -> dict[str, str]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError("GPU_QUOTA_EVIDENCE_INVALID", str(path)) from exc
    rows = [
        item
        for item in value
        if isinstance(item, Mapping) and str(item.get("resource", "")).upper() == "GPU"
    ]
    if len(rows) != 1:
        _fail("GPU_QUOTA_EVIDENCE_INVALID", "expected one GPU row")
    row = rows[0]
    state = {
        name: _text(row.get(name), f"GPU quota {name}", 100)
        for name in ("used", "remaining", "total", "refreshAt")
    }
    return state


def reconcile_pending_control(
    *,
    pending_path: str | Path,
    ledger_path: str | Path,
    workspace_root: str | Path,
    manifest_output_path: str | Path,
    report_output_path: str | Path,
    quota_before_path: str | Path,
    quota_after_path: str | Path,
) -> dict[str, Any]:
    from .evaluation import (
        ExactReport,
        canonical_report_core,
        validate_exact_report,
    )
    from .manifests import EvaluationManifest

    root = Path(workspace_root).resolve(strict=True)
    quota_before = _gpu_quota_state(quota_before_path)
    quota_after = _gpu_quota_state(quota_after_path)
    if canonical_json_bytes(quota_before) != canonical_json_bytes(quota_after):
        _fail("GPU_QUOTA_CHANGED", "CPU acceptance cannot be reconciled")
    quota_evidence_sha = sha256_bytes(
        canonical_json_bytes({"before": quota_before, "after": quota_after})
    )
    pending = load_pending_control(pending_path)
    ledger = Ledger(Path(ledger_path), root)
    events = ledger.read_events()
    controls = reconstruct_cpu_acceptances(events)
    state = controls.get(pending.run_id)
    if state is None:
        _fail("UNKNOWN_ACCEPTANCE_REQUEST", pending.run_id)
    if state.status in {CpuAcceptanceStatus.COMPLETED, CpuAcceptanceStatus.FAILED}:
        _fail("REQUEST_ALREADY_CONSUMED", pending.run_id)
    if state.status is not CpuAcceptanceStatus.RUNNING or state.started is None:
        _fail("ACCEPTANCE_REQUEST_NOT_STARTED", pending.run_id)
    registration_event = next(
        item
        for item in state.events
        if item.event_type is EventType.CPU_ACCEPTANCE_REGISTERED
    )
    if pending.registration_event_sha256 != event_sha256(registration_event):
        _fail("REQUEST_REGISTRATION_HASH_MISMATCH", pending.run_id)
    registered = state.registered
    if (
        pending.request_nonce != registered["request_nonce"]
        or pending.acceptance_request_sha256 != registered["acceptance_request_sha256"]
        or pending.evaluation_run_id != registered["evaluation_run_id"]
        or pending.kernel_ref != state.started["kernel_ref"]
        or pending.runtime_dataset_ref != state.started["runtime_dataset_ref"]
        or pending.runtime_bundle_name != registered["runtime_bundle_name"]
        or pending.runtime_bundle_sha256 != registered["runtime_bundle_sha256"]
        or pending.runtime_bundle_inventory_sha256
        != registered["runtime_bundle_inventory_sha256"]
        or pending.runtime_bundle_uncompressed_size_bytes
        != registered["runtime_bundle_uncompressed_size_bytes"]
        or pending.runtime_bundle_file_count
        != registered["runtime_bundle_file_count"]
    ):
        _fail("ACCEPTANCE_REQUEST_BINDING_MISMATCH", pending.run_id)
    identities = pending.value["source_identities"]
    registered_identity = {
        "scorer_lock_sha256": registered["scorer_lock_sha256"],
        "environment_lock_sha256": registered["environment_lock_sha256"],
        "manifest_policy_sha256": registered["manifest_policy_sha256"],
        "control_model_sha256": registered["control_model_sha256"],
        "config_sha256": registered["config_sha256"],
        "code_sha256": registered["code_sha256"],
        "data_source_sha256": registered["data_source_sha256"],
    }
    if canonical_json_bytes(identities) != canonical_json_bytes(registered_identity):
        _fail("ACCEPTANCE_SOURCE_MISMATCH", pending.run_id)
    manifest_value = pending.value["manifest"]
    manifest = EvaluationManifest.from_dict(manifest_value["manifest_document"])
    if (
        manifest.manifest_sha256 != manifest_value["manifest_sha256"]
        or len(manifest.samples) != manifest_value["sample_count"]
        or manifest.overlap_audit.get("passed") is not True
    ):
        _fail("MANIFEST_RECONCILIATION_MISMATCH", pending.run_id)
    fold_binding = [
        {
            "fold_id": fold.fold_id,
            "train_membership_sha256": fold.train_membership_sha256,
            "calibration_membership_sha256": fold.calibration_membership_sha256,
            "evaluation_membership_sha256": fold.evaluation_membership_sha256,
        }
        for fold in manifest.folds
    ]
    if canonical_json_bytes(fold_binding) != canonical_json_bytes(manifest_value["folds"]):
        _fail("MANIFEST_RECONCILIATION_MISMATCH", "fold memberships")
    binding_event = ExperimentEvent.create(
        pending.run_id,
        EventType.CPU_ACCEPTANCE_INPUTS_BOUND,
        cpu_acceptance_inputs_bound_payload(
            run_id=pending.run_id,
            manifest_sha256=manifest.manifest_sha256,
            folds=fold_binding,
        ),
    )
    ledger.append(binding_event)
    control = pending.value["control"]
    reconciliation_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "pending_payload_sha256": pending.pending_payload_sha256,
                "pending_envelope_sha256": pending.pending_envelope_sha256,
                "output_inventory_sha256": pending.output_inventory_sha256,
                "registration_event_sha256": event_sha256(registration_event),
                "start_event_sha256": event_sha256(
                    next(
                        item
                        for item in state.events
                        if item.event_type is EventType.CPU_ACCEPTANCE_STARTED
                    )
                ),
                "input_binding_event_sha256": event_sha256(binding_event),
                "gpu_quota_evidence_sha256": quota_evidence_sha,
            }
        )
    )
    completion_event = ExperimentEvent.create(
        pending.run_id,
        EventType.CPU_ACCEPTANCE_COMPLETED,
        cpu_acceptance_completed_payload(
            run_id=pending.run_id,
            actual_cpu_runtime_seconds=pending.value["actual_cpu_runtime_seconds"],
            peak_memory_mb=pending.value["peak_memory_mb"],
            remote_job_identity=pending.kernel_ref,
            graph_inventory_sha256=control["graph_inventory_sha256"],
            artifact_hashes=control["artifact_hashes"],
            output_inventory_sha256=pending.output_inventory_sha256,
            pending_payload_sha256=pending.pending_payload_sha256,
            pending_envelope_sha256=pending.pending_envelope_sha256,
            reconciliation_sha256=reconciliation_sha,
        ),
    )
    ledger.append(completion_event)
    events = ledger.read_events()
    members = [
        resolved_exact_member(
            events,
            role=role,
            fold_id=fold.fold_id,
            producer_run_id=pending.run_id,
        )
        for role in ("baseline", "candidate")
        for fold in manifest.folds
    ]
    members.sort(key=lambda item: (item["role"], item["fold_id"]))
    evaluation_policy_sha = _sha(
        control["evaluation_policy_sha256"], "control.evaluation_policy_sha256"
    )
    aggregate_registration = ExperimentEvent.create(
        pending.evaluation_run_id,
        EventType.EXACT_EVALUATION_REGISTERED,
        exact_evaluation_registration_payload(
            evaluation_run_id=pending.evaluation_run_id,
            scorer_lock_sha256=registered["scorer_lock_sha256"],
            environment_lock_sha256=registered["environment_lock_sha256"],
            manifest_sha256=manifest.manifest_sha256,
            evaluation_policy_sha256=evaluation_policy_sha,
            evidence_kind="official_data_control",
            members=members,
        ),
    )
    ledger.append(aggregate_registration)
    aggregate_start = ExperimentEvent.create(
        pending.evaluation_run_id,
        EventType.EXACT_EVALUATION_STARTED,
        exact_evaluation_started_payload(evaluation_run_id=pending.evaluation_run_id),
    )
    ledger.append(aggregate_start)
    authoritative = list(control["authoritative_inventories"])
    comparison = json.loads(json.dumps(control["comparison"]))
    comparison["member_binding_sha256"] = sha256_bytes(canonical_json_bytes(members))
    comparison["authoritative_inventory_binding_sha256"] = sha256_bytes(
        canonical_json_bytes(authoritative)
    )
    official = _rebind_official_producers(control["official"], members)
    registration = reconstruct_exact_evaluations(ledger.read_events())[
        pending.evaluation_run_id
    ].registered
    core = canonical_report_core(
        registration=registration,
        authoritative_inventories=authoritative,
        official=official,
        expected_samples=control["expected_sample_ids"],
        diagnostics=control["diagnostics"],
        comparison=comparison,
    )
    core_sha = sha256_bytes(canonical_json_bytes(core))
    envelope = {
        "schema_version": "biohub.exact-report-envelope.v1",
        "report_core_sha256": core_sha,
        "created_at": utc_now().isoformat().replace("+00:00", "Z"),
        "runtime_seconds": pending.value["actual_cpu_runtime_seconds"],
        "peak_memory_bytes": int(Decimal(pending.value["peak_memory_mb"]) * 1024 * 1024),
        "presentation_metadata": {
            "authority": "locally_reconciled_official_data_control",
            "cpu_registration_event_sha256": event_sha256(registration_event),
            "cpu_input_binding_event_sha256": event_sha256(binding_event),
            "cpu_terminal_event_sha256": event_sha256(completion_event),
            "aggregate_registration_event_sha256": event_sha256(
                aggregate_registration
            ),
            "pending_payload_sha256": pending.pending_payload_sha256,
            "reconciliation_sha256": reconciliation_sha,
            "gpu_quota_evidence_sha256": quota_evidence_sha,
            "authorized_for_promotion": False,
            "authorized_for_submission": False,
        },
    }
    envelope_sha = sha256_bytes(canonical_json_bytes(envelope))
    aggregate_completion = ExperimentEvent.create(
        pending.evaluation_run_id,
        EventType.EXACT_EVALUATION_COMPLETED,
        exact_evaluation_completed_payload(
            evaluation_run_id=pending.evaluation_run_id,
            members=members,
            report_core_sha256=core_sha,
            envelope_sha256=envelope_sha,
            artifact_hashes={
                "report_core": core_sha,
                "report_envelope": envelope_sha,
                "pending_payload": pending.pending_payload_sha256,
            },
            authoritative_inventories=authoritative,
            promotion_eligible=False,
        ),
    )
    ledger.append(aggregate_completion)
    report = ExactReport(core, envelope, core_sha, envelope_sha)
    validate_exact_report(
        report,
        ledger_path=ledger.path,
        workspace_root=root,
        require_completed=True,
    )
    accepted = {
        "schema_version": "biohub.phase2-control-acceptance.v1",
        "evidence_kind": "official_data_control",
        "accepted": True,
        "authorized_for_promotion": False,
        "authorized_for_submission": False,
        "cpu_run_id": pending.run_id,
        "evaluation_run_id": pending.evaluation_run_id,
        "kernel_ref": pending.kernel_ref,
        "runtime_dataset_ref": pending.runtime_dataset_ref,
        "runtime_bundle_name": pending.runtime_bundle_name,
        "runtime_bundle_sha256": pending.runtime_bundle_sha256,
        "runtime_bundle_inventory_sha256": pending.runtime_bundle_inventory_sha256,
        "runtime_bundle_uncompressed_size_bytes": (
            pending.runtime_bundle_uncompressed_size_bytes
        ),
        "runtime_bundle_file_count": pending.runtime_bundle_file_count,
        "accelerator": "none",
        "competition_submission_performed": False,
        "manifest_sha256": manifest.manifest_sha256,
        "report_core_sha256": core_sha,
        "report_envelope_sha256": envelope_sha,
        "pending_payload_sha256": pending.pending_payload_sha256,
        "pending_envelope_sha256": pending.pending_envelope_sha256,
        "output_inventory_sha256": pending.output_inventory_sha256,
        "reconciliation_sha256": reconciliation_sha,
        "gpu_quota_before": quota_before,
        "gpu_quota_after": quota_after,
        "gpu_quota_unchanged": True,
        "gpu_quota_evidence_sha256": quota_evidence_sha,
        "event_hashes": {
            "cpu_registered": event_sha256(registration_event),
            "cpu_inputs_bound": event_sha256(binding_event),
            "cpu_completed": event_sha256(completion_event),
            "aggregate_registered": event_sha256(aggregate_registration),
            "aggregate_started": event_sha256(aggregate_start),
            "aggregate_completed": event_sha256(aggregate_completion),
        },
        "exact_report": {"core": core, "envelope": envelope},
    }
    manifest_target = Path(manifest_output_path)
    report_target = Path(report_output_path)
    if manifest_target.exists() or report_target.exists():
        _fail("IMMUTABLE_ACCEPTANCE_OUTPUT_EXISTS", "manifest or report")
    atomic_write_json(manifest_target, manifest.to_dict())
    atomic_write_json(report_target, accepted)
    return accepted
