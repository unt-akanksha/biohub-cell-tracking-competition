from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET = STAGING_ROOT / "biohub-hoct-candidate-runtime-v1"
BASE_RUNTIME = STAGING_ROOT / "biohub-hoct-multibackbone-runtime-v1"
SOURCES = {
    "association_ensemble.py": ROOT / "research" / "association_ensemble.py",
    "biohub_adapter.py": ROOT / "research" / "hoct_graph" / "biohub_adapter.py",
    "multibackbone.py": ROOT / "research" / "hoct_graph" / "multibackbone.py",
    "train_biohub_hoct_probe.py": ROOT
    / "research"
    / "hoct_graph"
    / "train_biohub_hoct_probe.py",
    "rerank_hoct_multibackbone_submission.py": ROOT
    / "research"
    / "hoct_graph"
    / "rerank_hoct_multibackbone_submission.py",
}
EXPECTED_MODELS = {
    "general_v1.pt": "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33",
    "ctc_v0.pt": "b9be3d976e2d51ae946128ded99142a81b5ba99fb87a0da67c38de2934944000",
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


def checked_target() -> Path:
    root = STAGING_ROOT.resolve()
    target = TARGET.resolve()
    if target.parent != root or target.name != "biohub-hoct-candidate-runtime-v1":
        raise RuntimeError(f"Unsafe candidate-runtime target: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    target = checked_target()
    if target.exists():
        if not args.replace:
            raise FileExistsError(f"{target} already exists; pass --replace")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    license_path = BASE_RUNTIME / "HOCT_LICENSE"
    for name, expected_hash in EXPECTED_MODELS.items():
        model = BASE_RUNTIME / name
        if sha256_file(model) != expected_hash:
            raise RuntimeError(f"Pinned HOCT model is missing or changed: {name}")
        shutil.copy2(model, target / name)
    shutil.copy2(license_path, target / license_path.name)
    for name, source in SOURCES.items():
        shutil.copy2(source, target / name)

    files = {
        path.relative_to(target).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    write_json(
        target / "SOURCE_MANIFEST.json",
        {
            "schema_version": 1,
            "purpose": "Acceptance-bound HOCT submission inference; refuses public-edge replicas",
            "upstream": {
                "repository": "https://github.com/royerlab/hoct",
                "commit": "2ccc5040823bc944ab67790abd1f56eea7cd4f05",
                "license": "MIT",
            },
            "pretrained_models": {
                "general": {
                    "name": "general_v1",
                    "sha256": EXPECTED_MODELS["general_v1.pt"],
                    "parameters": 6252593,
                },
                "ctc": {
                    "name": "ctc_v0",
                    "sha256": EXPECTED_MODELS["ctc_v0.pt"],
                    "parameters": 6252593,
                },
            },
            "gate": {
                "requires_positive_clean_acceptance": True,
                "requires_hash_bound_probe": True,
                "rejects_edge_identical_public_replica": True,
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub HOCT Candidate Runtime v1",
            "id": "indarkarhana/biohub-hoct-candidate-runtime-v1",
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "files": len(files) + 2,
                "bytes": sum(
                    path.stat().st_size for path in target.rglob("*") if path.is_file()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
