from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
TARGETS = {
    "v1": STAGING_ROOT / "biohub-spatialdino-pu-runtime-v1",
    "v2": STAGING_ROOT / "biohub-spatialdino-pu-runtime-v2",
}
UPSTREAM = ROOT / ".biohub" / "cache" / "spatialdino"
CHECKPOINT = (
    ROOT / ".biohub" / "cache" / "models" / "spatialdino-vits8-step244999-backbone.pth"
)
EXPECTED_COMMIT = "ca3ab86b34430d963f12a3909baaeb9343c63b7d"
EXPECTED_CHECKPOINT_SHA256 = (
    "47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8"
)
SOURCES = {
    "data.py": ROOT / "research" / "spatialdino_detection" / "data.py",
    "density_calibration.py": ROOT / "research" / "density_calibration.py",
    "encoder.py": ROOT / "research" / "spatialdino_association" / "encoder.py",
    "evaluate_pretrained_detector.py": ROOT
    / "research"
    / "spotiflow_biohub"
    / "evaluate_pretrained_detector.py",
    "evaluate_spatialdino_pu_detector.py": ROOT
    / "research"
    / "spatialdino_detection"
    / "evaluate_pu_detector.py",
    "inference.py": ROOT / "research" / "spatialdino_detection" / "inference.py",
    "model.py": ROOT / "research" / "spatialdino_detection" / "model.py",
    "pu_targets.py": ROOT / "research" / "spotiflow_biohub" / "pu_targets.py",
    "public_teacher.py": ROOT / "research" / "spotiflow_biohub" / "public_teacher.py",
    "train_spatialdino_pu_detector.py": ROOT
    / "research"
    / "spatialdino_detection"
    / "train_pu_detector.py",
}
V2_SOURCES = {
    **SOURCES,
    "distillation.py": ROOT / "research" / "spatialdino_detection" / "distillation.py",
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


def checked_target(variant: str) -> Path:
    root = STAGING_ROOT.resolve()
    target = TARGETS[variant].resolve()
    if target.parent != root or target.name not in {
        "biohub-spatialdino-pu-runtime-v1",
        "biohub-spatialdino-pu-runtime-v2",
    }:
        raise RuntimeError(f"unsafe SpatialDINO PU staging target: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--variant", choices=sorted(TARGETS), default="v1")
    args = parser.parse_args()
    target = checked_target(args.variant)
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
    sources = V2_SOURCES if args.variant == "v2" else SOURCES
    for name, source in sources.items():
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
            "purpose": (
                "Independent SpatialDINO hybrid detector with selective soft-teacher distillation and clean validation; no submission"
                if args.variant == "v2"
                else "Independent SpatialDINO hybrid detector training and clean validation; no submission"
            ),
            "upstream": {
                "repository": "https://github.com/kirchhausenlab/spatialdino",
                "commit": commit,
                "license": "MIT",
            },
            "pretrained_model": {
                "name": "SpatialDINO ViT-S/8 step 244999 backbone",
                "sha256": EXPECTED_CHECKPOINT_SHA256,
                "parameters": 21_501_312,
            },
            "candidate": {
                "architecture": "SpatialDINO + raw 3D pyramid + UNETR decoder",
                "parameters": 29_521_225,
                "public_predictions_copied": False,
                "teachers_are_frozen_pseudo_label_sources_only": True,
                "validation_movies_excluded_from_training": 12,
                "selective_soft_distillation": args.variant == "v2",
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": f"Biohub SpatialDINO PU Runtime {args.variant}",
            "id": f"indarkarhana/biohub-spatialdino-pu-runtime-{args.variant}",
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
