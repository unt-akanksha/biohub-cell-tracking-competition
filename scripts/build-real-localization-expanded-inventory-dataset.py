#!/usr/bin/env python
"""Finalize the private train-only expanded localization inventory dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_ID = "indarkarhana/biohub-real-localization-expanded-inventory-v2"
RUN_ID = "competition-real-localization-expanded-inventory-v2"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    inventory_path = args.root / "expanded_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not (
        inventory.get("schema_version") == 1
        and inventory.get("status") == "complete"
        and inventory.get("run_id") == RUN_ID
        and inventory.get("expansion_policy", {}).get("roles_expanded")
        == ["optimization"]
        and inventory.get("expansion_policy", {}).get("selection_centers_changed")
        is False
        and inventory.get("expansion_policy", {}).get("sealed_audit_centers_changed")
        is False
        and inventory.get("competition_test_data_read") is False
        and inventory.get("public_leaderboard_used_for_selection") is False
        and inventory.get("authorized_for_submission") is False
    ):
        raise ValueError("expanded localization inventory is ineligible")
    metadata = {
        "title": "Biohub Real Localization Expanded Inventory v2",
        "id": DATASET_ID,
        "licenses": [{"name": "MIT"}],
        "isPrivate": True,
    }
    (args.root / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "inventory_sha256": sha256_file(inventory_path),
                "summary": inventory["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
