#!/usr/bin/env python
"""Package a verified Antelume peak-ranking checkpoint for Kaggle validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".biohub" / "cache" / "antelume-peak-rank-detector-v1"
ARCHIVE = STATE / "peak-rank-detector-results.tar.gz"
REPORT = STATE / "harvest-verification.json"
TARGET = ROOT / ".biohub" / "staging" / "biohub-peak-rank-validation-runtime-v1"
ARCHIVE_ROOT = "synthetic256-real-positive-temporal-peak-rank-v1"
EXPECTED_TARGET_NAME = "biohub-peak-rank-validation-runtime-v1"
DATASET_ID = "indarkarhana/biohub-peak-rank-validation-runtime-v1"
DATASET_TITLE = "Biohub Peak Rank Validation Runtime v1"
SOURCES = {
    "model.py": ROOT / "research" / "peak_rank_detection" / "model.py",
    "inference.py": ROOT / "research" / "peak_rank_detection" / "inference.py",
    "peak_association_bridge.py": ROOT
    / "research"
    / "peak_rank_detection"
    / "association_bridge.py",
    "lsm_association_bridge.py": ROOT
    / "research"
    / "lsm_fm_detection"
    / "association_bridge.py",
    "predict_with_official_linker.py": ROOT
    / "research"
    / "peak_rank_detection"
    / "predict_with_official_linker.py",
    "evaluate_peak_rank_detector.py": ROOT
    / "research"
    / "peak_rank_detection"
    / "evaluate_peak_rank_detector.py",
    "density_calibration.py": ROOT / "research" / "density_calibration.py",
    "evaluate_pretrained_detector.py": ROOT
    / "research"
    / "spotiflow_biohub"
    / "evaluate_pretrained_detector.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checked_target() -> Path:
    staging = (ROOT / ".biohub" / "staging").resolve()
    target = TARGET.resolve()
    if target.parent != staging or target.name != EXPECTED_TARGET_NAME:
        raise RuntimeError(f"unsafe peak-ranking staging target: {target}")
    return target


def archive_bytes(relative: str) -> bytes:
    expected = f"{ARCHIVE_ROOT}/{relative}"
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        member = members.get(expected)
        if member is None or not member.isfile() or member.issym() or member.islnk():
            raise ValueError(f"missing safe archive member: {expected}")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"cannot read archive member: {expected}")
        return stream.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    target = checked_target()
    if not ARCHIVE.is_file() or not REPORT.is_file():
        raise FileNotFoundError("verified peak-ranking harvest is incomplete")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not (
        report.get("status") == "verified"
        and report.get("accepted_for_kaggle_validation") is True
        and report.get("audit_passed") is True
        and report.get("parameter_count") == 38_381_478
        and report.get("archive_sha256") == sha256_file(ARCHIVE)
    ):
        raise ValueError("peak-ranking harvest is not eligible for Kaggle validation")
    checkpoint = archive_bytes("peak_rank_detector.pt")
    terminal = archive_bytes("terminal.json")
    terminal_payload = json.loads(terminal)
    if hashlib.sha256(checkpoint).hexdigest() != report.get("checkpoint_sha256"):
        raise ValueError("harvested checkpoint no longer matches verification report")
    if terminal_payload.get("checkpoint_sha256") != report.get("checkpoint_sha256"):
        raise ValueError("harvested terminal no longer matches verification report")
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} already exists; pass --replace")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name, source in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)
    (target / "peak_rank_detector.pt").write_bytes(checkpoint)
    (target / "training_terminal.json").write_bytes(terminal)
    files = {
        path.relative_to(target).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(target.iterdir())
        if path.is_file()
    }
    write_json(
        target / "SOURCE_MANIFEST.json",
        {
            "schema_version": 1,
            "purpose": (
                "Two-GPU clean held-out validation plus a dormant official-linker "
                "bridge; no submission generation"
            ),
            "architecture": "independent temporal 3D ConvNeXt U-Net peak ranker",
            "parameter_count": 38_381_478,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "training_archive_sha256": report["archive_sha256"],
            "training_run_id": terminal_payload["run_id"],
            "training_audit_passed": True,
            "competition_test_data_read_during_training": False,
            "public_notebook_weights_read_during_training": False,
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": DATASET_TITLE,
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "files": len(files) + 2,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
