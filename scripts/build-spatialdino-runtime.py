from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGET = STAGING_ROOT / "biohub-spatialdino-runtime-v1"
UPSTREAM = ROOT / ".biohub" / "cache" / "spatialdino"
CHECKPOINT = (
    ROOT / ".biohub" / "cache" / "models" / "spatialdino-vits8-step244999-backbone.pth"
)
EXPECTED_COMMIT = "ca3ab86b34430d963f12a3909baaeb9343c63b7d"
EXPECTED_CHECKPOINT_SHA256 = (
    "47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8"
)
SOURCES = {
    "appearance.py": ROOT / "research" / "spatialdino_association" / "appearance.py",
    "correction.py": ROOT / "research" / "spatialdino_association" / "correction.py",
    "encoder.py": ROOT / "research" / "spatialdino_association" / "encoder.py",
    "validate_appearance_correction.py": ROOT
    / "research"
    / "spatialdino_association"
    / "validate_appearance_correction.py",
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
    if target.parent != root or target.name != "biohub-spatialdino-runtime-v1":
        raise RuntimeError(f"unsafe SpatialDINO staging target: {target}")
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
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"SpatialDINO repository commit changed: {commit}")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("SpatialDINO checkpoint hash changed")
    for name, source in SOURCES.items():
        shutil.copy2(source, target / name)
    shutil.copy2(CHECKPOINT, target / "spatialdino_vits8_backbone.pth")
    shutil.copy2(UPSTREAM / "LICENSE", target / "SPATIALDINO_LICENSE")

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
            "purpose": "Frozen SpatialDINO appearance features for degree-preserving Biohub association validation; no submission",
            "upstream": {
                "repository": "https://github.com/kirchhausenlab/spatialdino",
                "commit": commit,
                "license": "MIT",
            },
            "pretrained_model": {
                "name": "SpatialDINO ViT-S/8 step 244999 backbone",
                "source": "upstream GUI default checkpoint",
                "sha256": EXPECTED_CHECKPOINT_SHA256,
                "parameters": 21_501_312,
                "patch_size": [8, 8, 8],
                "embedding_dimension": 384,
            },
            "adapter": {
                "architecture_equivalence": "bit-exact against upstream encoder on bound checkpoint",
                "correction": "disjoint two-edge swaps preserving every node and in/out degree",
                "selection": "two complete movies, one per embryo prefix",
                "acceptance": "two disjoint complete movies after configuration freeze",
                "public_leaderboard_used": False,
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub SpatialDINO Runtime v1",
            "id": "indarkarhana/biohub-spatialdino-runtime-v1",
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
