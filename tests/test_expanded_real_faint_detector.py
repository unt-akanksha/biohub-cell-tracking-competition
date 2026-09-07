import json
from pathlib import Path

import pytest

from research.peak_rank_detection import train_expanded_real_faint_detector as expanded


def test_expanded_member_uses_more_optimization_only() -> None:
    assert expanded.EXPECTED_REAL_ROLE_COUNTS == {
        "optimization": 480,
        "selection": 17,
        "sealed_audit": 14,
    }
    assert expanded.REAL_INVENTORY_SHA256 == (
        "a80c9028b21fdba1746bc2686dc5c64f0852e7954d1e758ba1357bcca5aa0edc"
    )
    assert "expanded-real-pu-faint" in expanded.RUN_ID


def test_expanded_manifest_keeps_roles_disjoint(tmp_path: Path, monkeypatch) -> None:
    roles = {"optimization": 480, "selection": 17, "sealed_audit": 14}
    rows = []
    for role, count in roles.items():
        for index in range(count):
            path = tmp_path / role / f"{role}-{index}.npz"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"{role}-{index}".encode())
            rows.append(
                {
                    "role": role,
                    "stem": f"{role}-stem-{index}",
                    "path": path.relative_to(tmp_path).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": expanded.base.sha256_file(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": expanded.REAL_MANIFEST_RUN_ID,
        "inventory_sha256": expanded.REAL_INVENTORY_SHA256,
        "source_mode": "kaggle_cpu_direct_competition_train_expanded_optimization_only",
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
        "excluded_final_probe_stems": list(expanded.base.EXPECTED_EXCLUDED_PROBE_STEMS),
        "files": rows,
    }
    manifest_path = tmp_path / "real_localization_shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = expanded.validate_expanded_real_manifest(
        tmp_path, expanded.base.sha256_file(manifest_path)
    )

    assert len(result["files"]) == 511
    rows[-1]["stem"] = rows[0]["stem"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        expanded.validate_expanded_real_manifest(
            tmp_path, expanded.base.sha256_file(manifest_path)
        )
