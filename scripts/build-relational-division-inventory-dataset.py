#!/usr/bin/env python
"""Stage the frozen relational division inventory as a private Kaggle dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "indarkarhana/biohub-relational-division-inventory-v3"
RUN_ID = "competition-relational-division-inventory-dataset-v3"
INVENTORY_RUN_ID = "competition-relational-division-inventory-v3"
INVENTORY_NAME = "relational_division_inventory_v3.json"
GENERATOR = ROOT / "research/build_competition_relational_division_inventory.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "complete"
        and payload.get("run_id") == INVENTORY_RUN_ID
        and len(payload.get("final_probe_stems", [])) == 4
        and payload.get("audit_opened") is False
        and payload.get("competition_train_data_read") is True
        and payload.get("competition_test_data_read") is False
        and payload.get("public_code_copied") is False
        and payload.get("public_predictions_copied") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and payload.get("authorized_for_submission") is False
        and summary.get("rows") == len(payload.get("examples", []))
        and int(summary.get("positives", -1)) >= 100
        and int(summary.get("hard_negatives", -1)) >= 2_000
        and int(summary.get("inference_eligible_positives", -1)) >= 50
        and int(summary.get("inference_eligible_hard_negatives", -1)) >= 100
    ):
        raise ValueError("relational division inventory is ineligible")
    return payload


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "RELATIONAL_DIVISION_INVENTORY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {INVENTORY_NAME, GENERATOR.name}
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUN_ID
        and set(manifest.get("files", {})) == expected
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
    ):
        raise ValueError("relational inventory dataset manifest is invalid")
    for name, record in manifest["files"].items():
        path = root / name
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"relational inventory dataset file changed: {name}")
    inventory = load_inventory(root / INVENTORY_NAME)
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "inventory_sha256": sha256_file(root / INVENTORY_NAME),
        "rows": inventory["summary"]["rows"],
        "authorized_for_cpu_extraction": True,
        "authorized_for_submission": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    load_inventory(args.inventory)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    sources = {
        INVENTORY_NAME: args.inventory,
        GENERATOR.name: GENERATOR,
    }
    for name, source in sources.items():
        shutil.copy2(source, args.output_root / name)
    write_json(
        args.output_root / "RELATIONAL_DIVISION_INVENTORY_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": RUN_ID,
            "files": {
                name: {"path": name, "sha256": sha256_file(args.output_root / name)}
                for name in sources
            },
            "competition_train_data_read": True,
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_command_included": False,
            "authorized_for_submission": False,
        },
    )
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub Relational Division Inventory v3",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_dataset(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
