from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify-relational-division-patch-archive.py"
SPEC = importlib.util.spec_from_file_location("relational_archive_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def build_archive(path: Path, *, include_extra: bool = False) -> None:
    payload = b"verified-shard"
    records = []
    embryos = ("44b6", "6bba")
    roles = ("optimization", "selection", "audit")
    for index in range(146):
        embryo = embryos[index % 2]
        role = roles[(index // 2) % 3]
        records.append(
            {
                "path": f"{role}/{embryo}/movie-{index:03d}.npz",
                "stem": f"{embryo}_{index:08x}",
                "embryo": embryo,
                "role": role,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": 2,
                "positives": 0,
                "hard_negatives": 2,
                "inference_eligible_positives": 0,
                "inference_eligible_hard_negatives": 0,
            }
        )
    for index in range(6):
        records[index]["positives"] = 1
        records[index]["hard_negatives"] = 1
        records[index]["inference_eligible_positives"] = 1
        records[index]["inference_eligible_hard_negatives"] = 1
    records[0]["positives"] += 128
    records[0]["hard_negatives"] += 2_593
    records[0]["inference_eligible_positives"] += 49
    records[0]["inference_eligible_hard_negatives"] += 159
    for record in records:
        record["rows"] = record["positives"] + record["hard_negatives"]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": verifier.RUN_ID,
        "inventory_sha256": verifier.INVENTORY_SHA256,
        "summary": dict(verifier.EXPECTED_SUMMARY),
        "records": records,
        "final_probe_stems": [
            "44b6_12dfb391",
            "44b6_267148e4",
            "6bba_062c8d37",
            "6bba_07e24132",
        ],
        "final_probe_movies_extracted": False,
        "audit_features_extracted": True,
        "audit_labels_scored": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    with tarfile.open(path, "w:gz") as archive:
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo(
            f"{verifier.ROOT}/relational_division_patch_manifest.json"
        )
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
        for record in records:
            info = tarfile.TarInfo(f"{verifier.ROOT}/{record['path']}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if include_extra:
            extra = tarfile.TarInfo(f"{verifier.ROOT}/unbound.txt")
            extra.size = 1
            archive.addfile(extra, io.BytesIO(b"x"))


def test_verified_archive_binds_every_expected_shard(tmp_path: Path) -> None:
    archive = tmp_path / "patches.tar.gz"
    build_archive(archive)

    result = verifier.verify_archive(archive)

    assert result["status"] == "verified"
    assert result["record_count"] == 146
    assert result["summary"] == verifier.EXPECTED_SUMMARY
    assert result["competition_test_data_read"] is False


def test_archive_rejects_unbound_file(tmp_path: Path) -> None:
    archive = tmp_path / "patches.tar.gz"
    build_archive(archive, include_extra=True)

    with pytest.raises(ValueError, match="unbound"):
        verifier.verify_archive(archive)
