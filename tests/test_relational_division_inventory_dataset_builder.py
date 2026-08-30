from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-relational-division-inventory-dataset.py"
MODULE = runpy.run_path(str(SCRIPT))


def write_inventory(path: Path) -> None:
    examples = [{} for _ in range(3_000)]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "run_id": MODULE["INVENTORY_RUN_ID"],
                "examples": examples,
                "summary": {
                    "rows": len(examples),
                    "positives": 134,
                    "hard_negatives": 2_866,
                    "inference_eligible_positives": 55,
                    "inference_eligible_hard_negatives": 165,
                },
                "final_probe_stems": ["a", "b", "c", "d"],
                "audit_opened": False,
                "competition_train_data_read": True,
                "competition_test_data_read": False,
                "public_code_copied": False,
                "public_predictions_copied": False,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
                "authorized_for_submission": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_builder_hash_binds_inventory_and_generator(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    output = tmp_path / "dataset"
    write_inventory(inventory)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--inventory",
            str(inventory),
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    verified = MODULE["verify_dataset"](output)
    metadata = json.loads((output / "dataset-metadata.json").read_text())
    assert verified["rows"] == 3_000
    assert verified["authorized_for_submission"] is False
    assert metadata["id"] == MODULE["DATASET_ID"]
    assert metadata["isPrivate"] is True


def test_verifier_rejects_inventory_drift(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    output = tmp_path / "dataset"
    write_inventory(inventory)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--inventory",
            str(inventory),
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (output / MODULE["INVENTORY_NAME"]).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file changed"):
        MODULE["verify_dataset"](output)
