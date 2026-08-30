#!/usr/bin/env python
"""Fail closed on a materialized train-only real-localization shard cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FINAL_PROBE_STEMS = [
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
]
EXPECTED_INVENTORY_SHA256 = (
    "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "real_localization_shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = manifest.get("files")
    by_role = manifest.get("summary", {}).get("by_role", {})
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == "competition-real-localization-shards-v1"
        and manifest.get("inventory_sha256") == EXPECTED_INVENTORY_SHA256
        and manifest.get("excluded_final_probe_stems") == FINAL_PROBE_STEMS
        and manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_created") is False
        and manifest.get("authorized_for_submission") is False
        and isinstance(files, list)
        and len(files) == 177
        and manifest.get("summary", {}).get("shards") == 177
        and by_role.get("optimization", {}).get("shards") == 146
        and by_role.get("selection", {}).get("shards") == 17
        and by_role.get("sealed_audit", {}).get("shards") == 14
    ):
        raise ValueError("real localization shard manifest is ineligible")
    seen: set[str] = set()
    total_bytes = 0
    for row in files:
        relative = str(row.get("path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
            or row.get("role") not in {
                "optimization",
                "selection",
                "sealed_audit",
            }
            or row.get("stem") in FINAL_PROBE_STEMS
            or relative_path.parts[0] != row.get("role")
        ):
            raise ValueError(f"invalid real shard record: {relative}")
        seen.add(relative)
        path = root / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"real localization shard changed: {relative}")
        total_bytes += path.stat().st_size
    if total_bytes != int(manifest["summary"]["bytes"]):
        raise ValueError("real localization shard byte summary changed")
    return {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": sha256_file(manifest_path),
        "source_mode": manifest.get("source_mode", "local_frame_cache"),
        "shards": len(files),
        "bytes": total_bytes,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
