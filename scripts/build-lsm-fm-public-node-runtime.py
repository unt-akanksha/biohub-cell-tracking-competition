from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-boundary-runtime.py"
DATASET_NAME = "biohub-lsm-fm-public-node-runtime-v1"
TARGET = ROOT / ".biohub" / "staging" / DATASET_NAME
EVALUATOR = (
    ROOT / "research" / "lsm_fm_detection" / "evaluate_public_node_refinement.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    values = runpy.run_path(str(BASE_BUILDER))
    base_main = values["main"]
    base_main.__globals__.update({"DATASET_NAME": DATASET_NAME, "TARGET": TARGET})
    original_argv = sys.argv
    try:
        sys.argv = [str(BASE_BUILDER)] + (["--replace"] if args.replace else [])
        base_main()
    finally:
        sys.argv = original_argv

    evaluator_name = EVALUATOR.name
    shutil.copy2(EVALUATOR, TARGET / evaluator_name)
    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["purpose"] = (
        "Clean LSM-FM coordinate refinement of frozen public-graph nodes; "
        "node IDs, counts, and edges preserved; no submission"
    )
    manifest["candidate"].update(
        {
            "architecture": "independent frozen 35.1M-parameter LSM-FM feature-36 detector",
            "coordinate_base": "frozen clean public graph",
            "node_ids_preserved": True,
            "node_count_preserved": True,
            "edges_preserved": True,
            "selection_movies": 8,
            "sealed_acceptance_movies": 4,
            "public_leaderboard_used_for_selection": False,
            "competition_submission_performed": False,
        }
    )
    path = TARGET / evaluator_name
    manifest["files"][evaluator_name] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    write_json(manifest_path, manifest)

    metadata_path = TARGET / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "title": "Biohub LSM-FM Public Node Runtime v1",
            "id": f"indarkarhana/{DATASET_NAME}",
            "description": (
                "Private hash-bound runtime for independent LSM-FM refinement of "
                "frozen public graph coordinates."
            ),
            "isPrivate": True,
        }
    )
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "target": str(TARGET),
                "manifest_sha256": sha256_file(manifest_path),
                "files": sum(path.is_file() for path in TARGET.rglob("*")),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
