#!/usr/bin/env python
"""Verify every member of the hard-negative real-division archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.temporal_contrastive.train_real_division_gate import sha256_file
from research.temporal_contrastive.train_real_division_gate_v2 import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.data_root / "real_division_hard_negative_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    roles_by_stem: dict[str, set[str]] = {}
    for record in manifest["shards"]:
        path = args.data_root / record["path"]
        if not (
            path.is_file()
            and path.stat().st_size == int(record["bytes"])
            and sha256_file(path) == record["sha256"]
        ):
            raise ValueError(f"hard-negative archive member changed: {record['path']}")
        roles_by_stem.setdefault(record["stem"], set()).add(record["role"])
    if any(len(roles) != 1 for roles in roles_by_stem.values()):
        raise RuntimeError("hard-negative movie roles overlap")
    print(
        json.dumps(
            {
                "status": "verified",
                "manifest_sha256": sha256_file(manifest_path),
                "shards": len(manifest["shards"]),
                "movies": len(roles_by_stem),
                "rows": manifest["summary"]["rows"],
                "division_positives": manifest["summary"]["division_positives"],
                "negative_frames": manifest["summary"]["negative_frames"],
                "audit_opened": manifest["audit_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
