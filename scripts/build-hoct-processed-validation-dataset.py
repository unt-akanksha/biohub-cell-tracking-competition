from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET = STAGING_ROOT / "biohub-hoct-processed-validation-v1"
SOURCE = ROOT / ".biohub" / "cache" / "kernel-outputs" / "hoct-multibackbone-probe-v1"
FILES = {
    "launcher_terminal.json": (
        SOURCE / "launcher_terminal.json",
        "dba40bccca19fab0c337ac9ad390e93a7dce47daed59095cd93223e394754e31",
    ),
    "training_terminal.json": (
        SOURCE / "hoct_multibackbone_v1" / "training_terminal.json",
        "8ddd87ca0dd748b668e8c4f61409930d071ef84f9dd82d96fd750e8b98943f69",
    ),
    "complete_movie_validation.json": (
        SOURCE / "hoct_multibackbone_v1" / "complete_movie_validation.json",
        "ea42d2976f4530684c38dc172b3fca5a70f1072a09dc52bdbe686eddafc53599",
    ),
    "processed_validation.csv": (
        SOURCE / "processed_validation" / "processed_validation.csv",
        "6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b",
    ),
    "processed_validation_report.json": (
        SOURCE / "processed_validation" / "processed_validation_report.json",
        "5a798804a0b56a081780b1d0213505e34d878c027d9e901dd3848eb9106db367",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    target = TARGET.resolve()
    if target.parent != STAGING_ROOT.resolve() or target.name != TARGET.name:
        raise RuntimeError(f"unsafe topology staging target: {target}")
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} already exists; pass --replace")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    copied = {}
    for name, (source, expected) in FILES.items():
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {name}: {actual}")
        shutil.copy2(source, target / name)
        copied[name] = {"bytes": source.stat().st_size, "sha256": actual}
    launcher = json.loads((target / "launcher_terminal.json").read_text(encoding="utf-8"))
    if not (
        launcher.get("run_id") == "hoct-multibackbone-probe-v1"
        and launcher.get("status") == "completed"
        and launcher.get("submission_created") is False
        and launcher.get("training_terminal_sha256") == FILES["training_terminal.json"][1]
    ):
        raise RuntimeError("HOCT producer terminal does not prove the expected clean output")
    write_json(
        target / "SOURCE_MANIFEST.json",
        {
            "schema_version": 1,
            "purpose": "Materialize hash-bound HOCT clean-validation output for downstream SpatialDINO association validation",
            "producer_run_id": "hoct-multibackbone-probe-v1",
            "producer_status": "completed",
            "producer_submission_created": False,
            "ground_truth_labels_in_processed_csv": False,
            "files": copied,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub HOCT Processed Validation v1",
            "id": "indarkarhana/biohub-hoct-processed-validation-v1",
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "files": len(copied) + 2,
                "bytes": sum(path.stat().st_size for path in target.iterdir() if path.is_file()),
            }
        )
    )


if __name__ == "__main__":
    main()
