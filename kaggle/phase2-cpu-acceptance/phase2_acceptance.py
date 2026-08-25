from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import struct
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path, PurePosixPath


# The local acceptance wrapper replaces this empty value only in the staged,
# owned kernel version. Keeping the request in the kernel artifact allows a
# corrective kernel-only version to bind a fresh ledger request without
# mutating the already-ready runtime dataset.
EMBEDDED_ACCEPTANCE_REQUEST_B64 = ""
BUNDLE_MAGIC = b"BIOHUB-RUNTIME-BUNDLE\x00\x01"
BUNDLE_SCHEMA = "biohub.runtime-bundle.v1"
BUNDLE_MAX_UNCOMPRESSED_BYTES = 2_147_483_648
BUNDLE_ALLOWED_ROOTS = frozenset(
    {"config", "kaggle", "requirements", "site-packages", "src", "tests", "vendor"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _one_with_marker(
    marker: str,
    input_root: Path = Path("/kaggle/input"),
    *,
    competition_slug: str | None = None,
) -> Path:
    """Find one Kaggle input mount in a supported bounded layout."""
    direct: list[Path] = []
    owner_qualified: list[Path] = []
    namespaced: list[Path] = []
    for first in sorted(input_root.iterdir()):
        if not first.is_dir():
            continue
        if first.name in {"competitions", "datasets"}:
            continue
        if (first / marker).exists():
            direct.append(first)
            # A direct source can contain large competition trees. It cannot
            # also be an owner namespace for this marker, so do not scan it.
            continue
        for second in sorted(first.iterdir()):
            if second.is_dir() and (second / marker).exists():
                owner_qualified.append(second)
    if competition_slug is not None:
        if "/" in competition_slug or not competition_slug.strip():
            raise RuntimeError("competition_slug_invalid")
        candidate = input_root / "competitions" / competition_slug
        if (candidate / marker).exists():
            namespaced.append(candidate)
    matches = [*direct, *owner_qualified, *namespaced]
    if len(matches) != 1:
        marker_name = Path(marker).name
        raise RuntimeError(
            "mounted_source_cardinality: "
            f"marker={marker_name} direct={len(direct)} "
            f"owner_qualified={len(owner_qualified)} "
            f"namespaced={len(namespaced)}"
        )
    return matches[0]


def _one_bundle(
    name: str,
    input_root: Path = Path("/kaggle/input"),
    *,
    dataset_slug: str | None = None,
) -> Path:
    if name != "biohub-runtime-v1.biohubbundle":
        raise RuntimeError(f"runtime_bundle_name_invalid: {name}")
    direct: list[Path] = []
    owner_qualified: list[Path] = []
    namespaced: list[Path] = []
    for first in sorted(input_root.iterdir()):
        if not first.is_dir():
            continue
        if first.name in {"competitions", "datasets"}:
            continue
        direct_candidate = first / name
        if direct_candidate.is_file():
            direct.append(direct_candidate)
            continue
        for second in sorted(first.iterdir()):
            candidate = second / name
            if second.is_dir() and candidate.is_file():
                owner_qualified.append(candidate)
    if dataset_slug is not None:
        parts = dataset_slug.split("/")
        if len(parts) != 2 or any(not part.strip() for part in parts):
            raise RuntimeError("runtime_dataset_slug_invalid")
        candidate = input_root / "datasets" / parts[0] / parts[1] / name
        if candidate.is_file():
            namespaced.append(candidate)
    matches = [*direct, *owner_qualified, *namespaced]
    if len(matches) != 1:
        raise RuntimeError(
            "runtime_bundle_cardinality: "
            f"name={name} direct={len(direct)} "
            f"owner_qualified={len(owner_qualified)} "
            f"namespaced={len(namespaced)}"
        )
    return matches[0]


def _acceptance_request(_runtime: Path | None = None) -> dict:
    if not EMBEDDED_ACCEPTANCE_REQUEST_B64:
        raise RuntimeError("embedded_acceptance_request_missing")
    raw = base64.b64decode(EMBEDDED_ACCEPTANCE_REQUEST_B64, validate=True)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("embedded_acceptance_request_invalid")
    return value


def _member_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise RuntimeError(f"runtime_bundle_path_invalid: {value}")
    path = PurePosixPath(value)
    parts = path.parts
    if (
        path.is_absolute()
        or len(parts) < 3
        or parts[0] != "bundle"
        or parts[1] not in BUNDLE_ALLOWED_ROOTS
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"runtime_bundle_path_invalid: {value}")
    return path


def _read_header(bundle: Path) -> tuple[dict, int]:
    with bundle.open("rb") as handle:
        if handle.read(len(BUNDLE_MAGIC)) != BUNDLE_MAGIC:
            raise RuntimeError("runtime_bundle_magic_invalid")
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise RuntimeError("runtime_bundle_header_invalid: length")
        length = struct.unpack(">Q", raw_length)[0]
        if length < 2 or length > 65_536:
            raise RuntimeError("runtime_bundle_header_invalid: length")
        header = json.loads(handle.read(length).decode("utf-8"))
        offset = handle.tell()
    fields = {
        "schema_version",
        "compression",
        "file_count",
        "uncompressed_size_bytes",
        "inventory_sha256",
        "payload_size_bytes",
        "payload_sha256",
        "tar_size_bytes",
    }
    if not isinstance(header, dict) or set(header) != fields:
        raise RuntimeError("runtime_bundle_header_invalid: fields")
    if header["schema_version"] != BUNDLE_SCHEMA or header["compression"] != "zlib":
        raise RuntimeError("runtime_bundle_header_invalid: schema")
    for name in (
        "file_count",
        "uncompressed_size_bytes",
        "payload_size_bytes",
        "tar_size_bytes",
    ):
        if not isinstance(header[name], int) or header[name] <= 0:
            raise RuntimeError(f"runtime_bundle_header_invalid: {name}")
    if (
        header["uncompressed_size_bytes"] > BUNDLE_MAX_UNCOMPRESSED_BYTES
        or bundle.stat().st_size - offset != header["payload_size_bytes"]
    ):
        raise RuntimeError("runtime_bundle_header_invalid: bounds")
    return header, offset


def _extract_bundle(bundle: Path, request: dict, destination: Path) -> Path:
    expected_sha = str(request["runtime_bundle_sha256"])
    if not secrets.compare_digest(_sha256_file(bundle), expected_sha):
        raise RuntimeError("runtime_bundle_hash_mismatch")
    header, offset = _read_header(bundle)
    expected = {
        "inventory_sha256": str(request["runtime_bundle_inventory_sha256"]),
        "uncompressed_size_bytes": int(
            request["runtime_bundle_uncompressed_size_bytes"]
        ),
        "file_count": int(request["runtime_bundle_file_count"]),
    }
    if any(header[name] != value for name, value in expected.items()):
        raise RuntimeError("runtime_bundle_inventory_mismatch: header")
    final = destination / "bundle"
    if final.exists():
        raise RuntimeError("runtime_bundle_destination_exists")
    temporary = Path(tempfile.mkdtemp(prefix=".biohub-extract-", dir=destination))
    tar_path = temporary / "payload.tar"
    try:
        payload_digest = hashlib.sha256()
        decompressor = zlib.decompressobj()
        expanded = 0
        with bundle.open("rb") as source, tar_path.open("xb") as target:
            source.seek(offset)
            while chunk := source.read(1024 * 1024):
                payload_digest.update(chunk)
                pending = chunk
                while pending:
                    decoded = decompressor.decompress(pending, 1024 * 1024)
                    pending = decompressor.unconsumed_tail
                    expanded += len(decoded)
                    if (
                        expanded > BUNDLE_MAX_UNCOMPRESSED_BYTES
                        or expanded > header["tar_size_bytes"]
                    ):
                        raise RuntimeError("runtime_bundle_expansion_limit")
                    target.write(decoded)
            tail = decompressor.flush()
            expanded += len(tail)
            target.write(tail)
        if (
            not decompressor.eof
            or decompressor.unused_data
            or expanded != header["tar_size_bytes"]
            or payload_digest.hexdigest() != header["payload_sha256"]
        ):
            raise RuntimeError("runtime_bundle_payload_invalid")
        records = []
        seen: set[str] = set()
        total = 0
        with tarfile.open(tar_path, "r:") as archive:
            for member in archive:
                if not member.isfile():
                    raise RuntimeError(
                        f"runtime_bundle_file_type_forbidden: {member.name}"
                    )
                relative = _member_path(member.name)
                folded = member.name.casefold()
                if folded in seen:
                    raise RuntimeError(f"runtime_bundle_duplicate_path: {member.name}")
                seen.add(folded)
                total += member.size
                if (
                    total > BUNDLE_MAX_UNCOMPRESSED_BYTES
                    or total > expected["uncompressed_size_bytes"]
                ):
                    raise RuntimeError("runtime_bundle_expansion_limit")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(
                        f"runtime_bundle_file_type_forbidden: {member.name}"
                    )
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                with target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > member.size:
                            raise RuntimeError(
                                f"runtime_bundle_member_size_invalid: {member.name}"
                            )
                        digest.update(chunk)
                        output.write(chunk)
                if written != member.size:
                    raise RuntimeError(
                        f"runtime_bundle_member_size_invalid: {member.name}"
                    )
                records.append(
                    {
                        "path": member.name,
                        "size_bytes": written,
                        "sha256": digest.hexdigest(),
                    }
                )
        records.sort(key=lambda item: item["path"])
        inventory_sha = hashlib.sha256(
            json.dumps(
                records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        actual = {
            "inventory_sha256": inventory_sha,
            "uncompressed_size_bytes": total,
            "file_count": len(records),
        }
        if actual != expected:
            raise RuntimeError("runtime_bundle_inventory_mismatch: extracted")
        tar_path.unlink()
        os.replace(temporary / "bundle", final)
        return destination
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}:
        raise RuntimeError("accelerator environment is not disabled")
    request = _acceptance_request()
    mounted_bundle = _one_bundle(
        str(request["runtime_bundle_name"]),
        dataset_slug=str(request["runtime_dataset_slug"]),
    )
    runtime = _extract_bundle(mounted_bundle, request, Path("/kaggle/working"))
    competition = _one_with_marker(
        "train", competition_slug=str(request["competition_slug"])
    )
    package_root = runtime / "bundle"
    sys.path[:0] = [
        str(package_root / "src"),
        str(package_root / "vendor" / "kaggle-cell-tracking-competition" / "src"),
        str(package_root / "vendor" / "tracksdata" / "src"),
        str(package_root / "site-packages"),
    ]
    from biohub_tracker.evaluation import evaluate_pending_control

    output_path = Path("/kaggle/working/pending-control-report.json")
    result = evaluate_pending_control(
        acceptance_request=request,
        competition_root=competition,
        runtime_root=runtime,
        output_path=output_path,
    )
    for generated in (runtime / "control-work", package_root):
        shutil.rmtree(generated)
        if generated.exists():
            raise RuntimeError(f"generated_output_cleanup_failed: {generated.name}")
    print(json.dumps({"status": result["status"], "output": output_path.name}))


if __name__ == "__main__":
    main()
