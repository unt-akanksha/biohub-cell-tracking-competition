from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-public-node-runtime.py"
DATASET_NAME = "biohub-lsm-fm-public-node-runtime-v2"
TARGET = ROOT / ".biohub" / "staging" / DATASET_NAME


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    values = runpy.run_path(str(BASE_BUILDER))
    base_main = values["main"]
    base_main.__globals__.update({"DATASET_NAME": DATASET_NAME, "TARGET": TARGET})
    original_argv = sys.argv
    try:
        base_main()
    finally:
        sys.argv = original_argv

    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["purpose"] = (
        "Selection-free acceptance of one predeclared LSM-FM coordinate "
        "refinement on frozen public-graph nodes; no submission"
    )
    manifest["candidate"].update(
        {
            "selection_movies": 0,
            "sealed_acceptance_movies": 4,
            "predeclared_strategy": "public_lsm_r2_p2_b025",
            "hyperparameter_selection_performed": False,
            "public_graph_used_as_base": True,
            "public_predictions_copied": True,
            "exact_public_replica": False,
        }
    )
    write_json(manifest_path, manifest)

    metadata_path = TARGET / "dataset-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "id": f"indarkarhana/{DATASET_NAME}",
            "title": "Biohub LSM-FM Public Node Runtime v2",
            "description": (
                "Private hash-bound runtime for selection-free acceptance of "
                "one predeclared LSM-FM public-node coordinate refinement."
            ),
        }
    )
    write_json(metadata_path, metadata)
    print(manifest_path)


if __name__ == "__main__":
    main()
