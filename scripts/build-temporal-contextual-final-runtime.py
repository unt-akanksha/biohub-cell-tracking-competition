#!/usr/bin/env python
"""Build a frozen final-only runtime with transition-balanced two-GPU inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / ".biohub"
    / "cache"
    / "dataset-redownloads"
    / "biohub-temporal-contextual-transfer-runtime-v1-version4"
)
TARGET = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-temporal-contextual-final-runtime-v1"
)
INFERENCE_SOURCE = (
    ROOT
    / "research"
    / "temporal_contrastive"
    / "dual_fold_appearance_submission.py"
)
SOURCE_MANIFEST_SHA256 = (
    "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
)
DATASET_ID = "indarkarhana/biohub-temporal-contextual-final-runtime-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checked_target() -> Path:
    staging = (ROOT / ".biohub" / "staging").resolve()
    target = TARGET.resolve()
    if target.parent != staging or target.name != TARGET.name:
        raise RuntimeError(f"unsafe final runtime target: {target}")
    return target


def build(*, replace: bool) -> dict:
    source = SOURCE.resolve()
    target = checked_target()
    manifest_path = source / "SOURCE_MANIFEST.json"
    if not source.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"frozen contextual runtime is missing: {source}")
    observed_manifest = sha256_file(manifest_path)
    if observed_manifest != SOURCE_MANIFEST_SHA256:
        raise RuntimeError(f"frozen contextual runtime changed: {observed_manifest}")
    if not INFERENCE_SOURCE.is_file():
        raise FileNotFoundError(f"final inference source is missing: {INFERENCE_SOURCE}")
    if target.exists():
        if not replace:
            raise FileExistsError(f"{target} exists; pass --replace to rebuild")
        shutil.rmtree(target)
    shutil.copytree(source, target)
    shutil.copy2(INFERENCE_SOURCE, target / INFERENCE_SOURCE.name)

    manifest = json.loads((target / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    inference = target / INFERENCE_SOURCE.name
    manifest["runtime_family"] = "contextual_final_transition_sharded_v1"
    manifest["purpose"] = (
        "frozen contextual-v3 final inference with exact transition partitioning "
        "across two GPUs; no training, leaderboard selection, or submit command"
    )
    manifest["parent_runtime"] = {
        "dataset_id": "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
        "dataset_version": 4,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "files_changed": [INFERENCE_SOURCE.name],
    }
    manifest["integrity"].update(
        {
            "transition_partitioned_inference": True,
            "transition_partition_kind": "dominant_movie_transition_split_v1",
            "transition_partition_prediction_equivalent": True,
            "each_consecutive_transition_processed_exactly_once": True,
        }
    )
    manifest["files"][INFERENCE_SOURCE.name] = {
        "bytes": inference.stat().st_size,
        "sha256": sha256_file(inference),
    }
    write_json(target / "SOURCE_MANIFEST.json", manifest)
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub Temporal Contextual Final Runtime v1",
            "id": DATASET_ID,
            "licenses": [{"name": "BSD-3-Clause"}],
            "isPrivate": True,
        },
    )
    return {
        "schema_version": 1,
        "status": "built",
        "target": str(target),
        "dataset_id": DATASET_ID,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "manifest_sha256": sha256_file(target / "SOURCE_MANIFEST.json"),
        "inference_sha256": sha256_file(inference),
        "transition_partitioned_inference": True,
        "competition_submission_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(replace=args.replace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
