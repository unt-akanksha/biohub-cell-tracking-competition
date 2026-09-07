#!/usr/bin/env python
"""Train the capacity detector on expanded real temporal coverage plus fading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.peak_rank_detection import train_synthetic_real_detector as base

try:
    from train_faint_cell_pu_detector import augment_example, training_loss
except ModuleNotFoundError:
    from research.peak_rank_detection.train_faint_cell_pu_detector import (
        augment_example,
        training_loss,
    )


RUN_ID = "synthetic256-expanded-real-pu-faint-temporal-peak-rank-v7"
REAL_MANIFEST_RUN_ID = "competition-real-localization-expanded-shards-v2"
REAL_INVENTORY_SHA256 = "a80c9028b21fdba1746bc2686dc5c64f0852e7954d1e758ba1357bcca5aa0edc"
EXPECTED_REAL_ROLE_COUNTS = {
    "optimization": 480,
    "selection": 17,
    "sealed_audit": 14,
}


def validate_expanded_real_manifest(root: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = root / "real_localization_shard_manifest.json"
    if base.sha256_file(manifest_path) != expected_sha256.lower():
        raise ValueError("expanded real localization manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == REAL_MANIFEST_RUN_ID
        and manifest.get("inventory_sha256") == REAL_INVENTORY_SHA256
        and manifest.get("source_mode")
        == "kaggle_cpu_direct_competition_train_expanded_optimization_only"
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
        and manifest.get("authorized_for_submission") is False
        and tuple(manifest.get("excluded_final_probe_stems", ()))
        == base.EXPECTED_EXCLUDED_PROBE_STEMS
    ):
        raise ValueError("expanded real localization manifest violates clean policy")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("expanded real localization manifest has no file inventory")
    for role, expected_count in EXPECTED_REAL_ROLE_COUNTS.items():
        role_rows = [row for row in rows if row.get("role") == role]
        if len(role_rows) != expected_count:
            raise ValueError(f"unexpected expanded {role} shard count")
    role_stems = {
        role: {str(row["stem"]) for row in rows if row.get("role") == role}
        for role in EXPECTED_REAL_ROLE_COUNTS
    }
    for left_index, left in enumerate(role_stems):
        for right in tuple(role_stems)[left_index + 1 :]:
            if role_stems[left] & role_stems[right]:
                raise ValueError(f"expanded real stems overlap between {left} and {right}")
    for row in rows:
        path = root / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"expanded real localization shard changed: {path}")
        if row["role"] != "sealed_audit" and base.sha256_file(path) != row["sha256"]:
            raise ValueError(f"expanded real localization shard hash changed: {path}")
    return manifest


def main() -> None:
    base.RUN_ID = RUN_ID
    base.validate_real_manifest = validate_expanded_real_manifest
    base.augment_example = augment_example
    base.training_loss = training_loss
    base.main()


if __name__ == "__main__":
    main()
