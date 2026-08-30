from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kaggle/biohub-relational-division-patches-v3/extract.py"
MODULE = runpy.run_path(str(SCRIPT))


def valid_inventory() -> dict:
    role_record = {
        "positives": 1,
        "hard_negatives": 1,
        "inference_eligible_positives": 1,
        "inference_eligible_hard_negatives": 1,
    }
    example = {
        "stem": "44b6_example",
        "embryo": "44b6",
        "role": "optimization",
        "timepoint": 4,
        "required_frames": [3, 4, 5],
        "parent_id": 1,
        "existing_child_id": 2,
        "proposed_child_id": 3,
        "centers_zyx_voxel": {
            "parent": [1.0, 2.0, 3.0],
            "existing_child": [1.0, 2.0, 2.0],
            "proposed_child": [1.0, 2.0, 4.0],
        },
        "daughter_order_invariant": True,
        "division_recovery_target": True,
        "inference_geometry_eligible": True,
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "run_id": MODULE["INVENTORY_RUN_ID"],
        "examples": [example] * 3_013,
        "summary": {
            "rows": 3_013,
            "positives": 134,
            "hard_negatives": 2_879,
            "inference_eligible_positives": 55,
            "inference_eligible_hard_negatives": 165,
            "by_embryo_role": {
                embryo: {
                    role: dict(role_record)
                    for role in ("optimization", "selection", "audit")
                }
                for embryo in ("44b6", "6bba")
            },
        },
        "final_probe_stems": sorted(MODULE["FINAL_PROBE_STEMS"]),
        "audit_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def test_inventory_contract_rejects_final_probe_or_opened_audit() -> None:
    payload = valid_inventory()
    payload["audit_opened"] = True

    with pytest.raises(ValueError, match="inventory changed"):
        MODULE["validate_inventory"](payload)


def test_shard_preserves_triplet_layout_geometry_and_eligibility(tmp_path: Path) -> None:
    examples = [
        {
            "division_recovery_target": True,
            "inference_geometry_eligible": True,
            "parent_id": 1,
            "existing_child_id": 2,
            "proposed_child_id": 3,
            **{name: 1.0 for name in MODULE["GEOMETRY_NAMES"]},
        },
        {
            "division_recovery_target": False,
            "inference_geometry_eligible": False,
            "parent_id": 4,
            "existing_child_id": 5,
            "proposed_child_id": 6,
            **{name: None for name in MODULE["GEOMETRY_NAMES"]},
        },
    ]
    patches = torch.zeros((2, 3, 3, 17, 17, 17), dtype=torch.float32)

    record = MODULE["write_shard"](
        tmp_path,
        stem="44b6_example",
        role="optimization",
        embryo="44b6",
        timepoint=4,
        examples=examples,
        patches=patches,
    )

    with np.load(tmp_path / record["path"], allow_pickle=False) as data:
        assert data["relational_patches"].shape == (2, 3, 3, 17, 17, 17)
        assert data["geometry_features"].shape == (2, len(MODULE["GEOMETRY_NAMES"]))
        assert data["label_weight"].tolist() == [1.0, 0.5]
        assert data["inference_geometry_eligible"].tolist() == [True, False]
        metadata = json.loads(str(data["metadata_json"].item()))
    assert metadata["daughter_order_invariant"] is True
    assert metadata["audit_labels_scored"] is False


def test_kernel_is_cpu_offline_and_submission_free() -> None:
    metadata = json.loads((SCRIPT.parent / "kernel-metadata.json").read_text())
    source = SCRIPT.read_text(encoding="utf-8")

    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert (
        "indarkarhana/biohub-relational-division-inventory-v3"
        in metadata["dataset_sources"]
    )
    assert "competition_test_data_read\": False" in source
    assert "kaggle competitions submit" not in source
