#!/usr/bin/env python
"""Build a CPU Kaggle extractor for expanded train-only detector replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "scripts" / "build-real-localization-replay-cache-kernel.py"))
KERNEL_ID = "biohub-real-localization-expanded-replay-cache-v2"
KERNEL_REF = f"indarkarhana/{KERNEL_ID}"
LABEL_DATASET_REF = "indarkarhana/biohub-real-localization-labels-v2"
INVENTORY_DATASET_REF = "indarkarhana/biohub-real-localization-expanded-inventory-v2"
INVENTORY_SLUG = "biohub-real-localization-expanded-inventory-v2"
PARENT_INVENTORY_SHA256 = "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expanded extractor source contract changed: {old!r}")
    return source.replace(old, new, 1)


def expanded_kernel_source(
    *,
    labels_manifest_sha256: str,
    support_wheel_name: str,
    support_wheel_sha256: str,
    inventory_path: Path,
) -> str:
    inventory_sha256 = sha256_file(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary = inventory["summary"]
    by_role = summary["by_role"]
    source = BASE["kernel_source"](
        labels_manifest_sha256=labels_manifest_sha256,
        support_wheel_name=support_wheel_name,
        support_wheel_sha256=support_wheel_sha256,
    )
    source = replace_once(
        source,
        'RUN_ID = "competition-real-localization-shards-v1"',
        'RUN_ID = "competition-real-localization-expanded-shards-v2"',
    )
    source = replace_once(
        source,
        f'EXPECTED_INVENTORY_SHA256 = "{PARENT_INVENTORY_SHA256}"',
        f'EXPECTED_PARENT_INVENTORY_SHA256 = "{PARENT_INVENTORY_SHA256}"\nEXPECTED_INVENTORY_SHA256 = "{inventory_sha256}"',
    )
    source = replace_once(
        source,
        'OUTPUT_ROOT = WORKING / "competition_real_localization_shards_v1"',
        'OUTPUT_ROOT = WORKING / "competition_real_localization_expanded_shards_v2"',
    )
    source = replace_once(
        source,
        'inventory_path = label_root / "inventory.json"',
        f'inventory_path = INPUT_ROOT / "{INVENTORY_SLUG}" / "expanded_inventory.json"',
    )
    source = replace_once(
        source,
        'labels_manifest.get("inventory_sha256") == EXPECTED_INVENTORY_SHA256',
        'labels_manifest.get("inventory_sha256") == EXPECTED_PARENT_INVENTORY_SHA256',
    )
    source = replace_once(
        source,
        'and inventory.get("status") == "complete"',
        'and inventory.get("status") == "complete"\n    and inventory.get("run_id") == "competition-real-localization-expanded-inventory-v2"\n    and inventory.get("parent_inventory_sha256") == EXPECTED_PARENT_INVENTORY_SHA256\n    and inventory.get("expansion_policy", {}).get("roles_expanded") == ["optimization"]\n    and inventory.get("expansion_policy", {}).get("selection_centers_changed") is False\n    and inventory.get("expansion_policy", {}).get("sealed_audit_centers_changed") is False',
    )
    source = replace_once(
        source,
        "len(records) != 177",
        f"len(records) != {int(summary['center_frames'])}",
    )
    source = replace_once(
        source,
        "len(frame_records) != 525",
        f"len(frame_records) != {int(summary['required_frames'])}",
    )
    for role in ("optimization", "selection", "sealed_audit"):
        old_count = {"optimization": 146, "selection": 17, "sealed_audit": 14}[role]
        source = replace_once(
            source,
            f'counts["{role}"]["shards"] != {old_count}',
            f'counts["{role}"]["shards"] != {int(by_role[role]["center_frames"])}',
        )
    source = replace_once(
        source,
        '"source_mode": "kaggle_cpu_direct_competition_train"',
        '"source_mode": "kaggle_cpu_direct_competition_train_expanded_optimization_only"',
    )
    source = replace_once(
        source,
        '"run_id": "competition-real-localization-kaggle-source-digests-v1"',
        '"run_id": "competition-real-localization-kaggle-source-digests-v2"',
    )
    source = replace_once(
        source,
        '"run_id": "competition-real-localization-kaggle-cache-v1"',
        '"run_id": "competition-real-localization-expanded-kaggle-cache-v2"',
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-upload-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "kaggle" / KERNEL_ID)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    upload = json.loads((args.label_upload_root / "upload_manifest.json").read_text())
    support = upload["support_wheel"]
    inventory_path = args.inventory_root / "expanded_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not (
        upload.get("status") == "complete"
        and upload.get("dataset_id") == LABEL_DATASET_REF
        and upload.get("inventory_sha256") == PARENT_INVENTORY_SHA256
        and inventory.get("run_id") == "competition-real-localization-expanded-inventory-v2"
        and inventory.get("competition_test_data_read") is False
        and inventory.get("authorized_for_submission") is False
    ):
        raise ValueError("expanded replay inputs are ineligible")
    source = expanded_kernel_source(
        labels_manifest_sha256=upload["labels_manifest_sha256"],
        support_wheel_name=support["path"],
        support_wheel_sha256=support["sha256"],
        inventory_path=inventory_path,
    )
    if args.output_root.exists():
        expected = {"kernel-metadata.json", f"{KERNEL_ID}.ipynb"}
        actual = {path.name for path in args.output_root.iterdir()}
        if not args.replace or actual != expected:
            raise FileExistsError("expanded replay kernel root is not safely replaceable")
    else:
        args.output_root.mkdir(parents=True)
    metadata = {
        "id": KERNEL_REF,
        "title": "Biohub Real Localization Expanded Replay Cache v2",
        "code_file": f"{KERNEL_ID}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["cpu", "cell-tracking", "training-data", "expanded-replay"],
        "dataset_sources": [LABEL_DATASET_REF, INVENTORY_DATASET_REF],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
    }
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {
                "accelerator": "none",
                "dataSources": [
                    {"sourceId": LABEL_DATASET_REF, "sourceType": "dataset"},
                    {"sourceId": INVENTORY_DATASET_REF, "sourceType": "dataset"},
                    {"sourceId": "biohub-cell-tracking-during-development", "sourceType": "competition"},
                ],
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [BASE["code_cell"](source)],
    }
    (args.output_root / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / f"{KERNEL_ID}.ipynb").write_text(
        json.dumps(notebook, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
