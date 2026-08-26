from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET = STAGING_ROOT / "biohub-hoct-multibackbone-runtime-v1"
HOCT_REPOSITORY = ROOT / ".biohub" / "cache" / "research" / "hoct"
MODEL_ROOT = ROOT / ".biohub" / "cache" / "models" / "hoct"
MODELS = {
    "general_v1.pt": (
        MODEL_ROOT / "general_v1.pt",
        "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33",
    ),
    "ctc_v0.pt": (
        MODEL_ROOT / "ctc_v0.pt",
        "b9be3d976e2d51ae946128ded99142a81b5ba99fb87a0da67c38de2934944000",
    ),
}
SOURCES = {
    "association_ensemble.py": ROOT / "research" / "association_ensemble.py",
    "biohub_adapter.py": ROOT / "research" / "hoct_graph" / "biohub_adapter.py",
    "multibackbone.py": ROOT / "research" / "hoct_graph" / "multibackbone.py",
    "train_biohub_hoct_probe.py": ROOT
    / "research"
    / "hoct_graph"
    / "train_biohub_hoct_probe.py",
    "train_biohub_hoct_multibackbone.py": ROOT
    / "research"
    / "hoct_graph"
    / "train_biohub_hoct_multibackbone.py",
}
EXPECTED_REPOSITORY_COMMIT = "2ccc5040823bc944ab67790abd1f56eea7cd4f05"


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
    if target.parent != root or target.name != "biohub-hoct-multibackbone-runtime-v1":
        raise RuntimeError(f"Unsafe HOCT V2 staging target: {target}")
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
    for name, (source, expected_hash) in MODELS.items():
        if sha256_file(source) != expected_hash:
            raise RuntimeError(f"Official HOCT checkpoint hash changed: {name}")
        shutil.copy2(source, target / name)
    for name, source in SOURCES.items():
        shutil.copy2(source, target / name)
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
            "purpose": "Biohub-supervised dual-checkpoint HOCT probe and clean acceptance; no submission",
            "upstream": {
                "repository": "https://github.com/royerlab/hoct",
                "commit": commit,
                "license": "MIT",
                "paper": "https://arxiv.org/abs/2607.11754",
            },
            "pretrained_models": {
                "general": {
                    "name": "general_v1",
                    "sha256": MODELS["general_v1.pt"][1],
                    "parameters": 6_252_593,
                    "edge_feature_dimension": 288,
                },
                "ctc": {
                    "name": "ctc_v0",
                    "sha256": MODELS["ctc_v0.pt"][1],
                    "parameters": 6_252_593,
                    "edge_feature_dimension": 288,
                },
            },
            "adaptation": {
                "learned_parameters_per_backbone": 289,
                "single_variants": 4,
                "support_aware_blend_variants": 6,
                "selection": "two held-out complete movies",
                "acceptance": "two disjoint complete movies inferred and scored once after freeze",
                "public_leaderboard_used": False,
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub HOCT Multibackbone Runtime v1",
            "id": "indarkarhana/biohub-hoct-multibackbone-runtime-v1",
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
                    path.stat().st_size
                    for path in target.rglob("*")
                    if path.is_file()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
