from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET = STAGING_ROOT / "biohub-hoct-runtime-v1"
HOCT_REPOSITORY = ROOT / ".biohub" / "cache" / "research" / "hoct"
HOCT_MODEL = ROOT / ".biohub" / "cache" / "models" / "hoct" / "general_v1.pt"
SOURCES = {
    "biohub_adapter.py": ROOT / "research" / "hoct_graph" / "biohub_adapter.py",
    "train_biohub_hoct_probe.py": ROOT
    / "research"
    / "hoct_graph"
    / "train_biohub_hoct_probe.py",
}
EXPECTED_REPOSITORY_COMMIT = "2ccc5040823bc944ab67790abd1f56eea7cd4f05"
EXPECTED_MODEL_SHA256 = "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33"


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
    if target.parent != root or target.name != "biohub-hoct-runtime-v1":
        raise RuntimeError(f"Unsafe HOCT staging target: {target}")
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

    commit = subprocess.check_output(
        ["git", "-C", str(HOCT_REPOSITORY), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_REPOSITORY_COMMIT:
        raise RuntimeError(f"HOCT repository commit changed: {commit}")
    if sha256_file(HOCT_MODEL) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Official HOCT general_v1 checkpoint hash changed")
    for name, source in SOURCES.items():
        shutil.copy2(source, target / name)
    shutil.copy2(HOCT_MODEL, target / "general_v1.pt")
    shutil.copy2(HOCT_REPOSITORY / "LICENSE", target / "HOCT_LICENSE")

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
            "purpose": "Biohub-supervised HOCT frozen-feature association probe; no public output replica",
            "upstream": {
                "repository": "https://github.com/royerlab/hoct",
                "commit": commit,
                "license": "MIT",
                "paper": "https://arxiv.org/abs/2607.11754",
            },
            "pretrained_model": {
                "name": "general_v1",
                "source": "official royerlab/hoct weights-v1 GitHub release",
                "sha256": EXPECTED_MODEL_SHA256,
                "parameters": 6_252_593,
                "edge_feature_dimension": 288,
            },
            "adaptation": {
                "learned_parameters": 289,
                "selection": "two held-out complete movies",
                "acceptance": "two disjoint complete movies scored once after freeze",
                "public_leaderboard_used": False,
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub HOCT Runtime v1",
            "id": "indarkarhana/biohub-hoct-runtime-v1",
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
