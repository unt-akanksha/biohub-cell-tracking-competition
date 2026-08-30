#!/usr/bin/env python
"""Stage the pinned CPU extractor base as a private Kaggle dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


DATASET_ID = "indarkarhana/biohub-real-division-extractor-runtime-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    target = args.output_root / "extract_base.py"
    shutil.copy2(args.source, target)
    (args.output_root / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Biohub Real Division Extractor Runtime v1",
                "id": DATASET_ID,
                "licenses": [{"name": "MIT"}],
                "isPrivate": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_root / "RUNTIME_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "source_sha256": sha256_file(target),
                "competition_data_included": False,
                "submission_command_included": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
