from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / ".biohub" / "staging"
DATASET_NAME = "biohub-lsm-fm-image-text-pu-runtime-v1"
TARGET = STAGING_ROOT / DATASET_NAME
CHECKPOINT = (
    ROOT
    / ".biohub"
    / "cache"
    / "models"
    / "lsm-fm"
    / "lsm_fm_image_text_student.pt"
)
MONAI_ROOT = ROOT / ".biohub" / "cache" / "spatialdino-deps"
EXPECTED_CHECKPOINT_SHA256 = "aca3c5d43ef7f3d7ed2ff169d1ab72b71a03acec293a48283d73d38fcf3520e7"
EXPECTED_SOURCE_SHA256 = "f08122ee471fae0c555f3b71df37087d6dd78ffda7bf000f0510aa262c1c5d15"
SOURCES = {
    "data.py": ROOT / "research" / "spatialdino_detection" / "data.py",
    "density_calibration.py": ROOT / "research" / "density_calibration.py",
    "distillation.py": ROOT / "research" / "spatialdino_detection" / "distillation.py",
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
    "lsm_fm_model.py": ROOT
    / "research"
    / "lsm_fm_detection"
    / "image_text_model.py",
    "model.py": ROOT / "research" / "spatialdino_detection" / "model.py",
    "pu_targets.py": ROOT / "research" / "spotiflow_biohub" / "pu_targets.py",
    "public_teacher.py": ROOT / "research" / "spotiflow_biohub" / "public_teacher.py",
    "train_spatialdino_pu_detector.py": ROOT
    / "research"
    / "spatialdino_detection"
    / "train_pu_detector.py",
    "LSM_FM_WEIGHTS_ATTRIBUTION.md": ROOT / "licenses" / "LSM_FM_WEIGHTS_ATTRIBUTION.md",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checked_target() -> Path:
    root = STAGING_ROOT.resolve()
    target = TARGET.resolve()
    if target.parent != root or target.name != DATASET_NAME:
        raise RuntimeError(f"unsafe multimodal LSM-FM staging target: {target}")
    return target


def build_monai_archive(source: Path, output: Path) -> dict[str, dict[str, int | str]]:
    """Write an importable, reproducible pure-Python MONAI archive."""

    inventory: dict[str, dict[str, int | str]] = {}
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(source).as_posix()
            payload = path.read_bytes()
            info = zipfile.ZipInfo(f"monai/{relative}", (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload, compresslevel=9)
            inventory[f"monai/{relative}"] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    return inventory


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

    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("stripped multimodal LSM-FM checkpoint hash changed")
    for name, source in SOURCES.items():
        shutil.copy2(source, target / name)
    shutil.copy2(CHECKPOINT, target / "lsm_fm_image_text_student.pt")
    monai_files = build_monai_archive(
        MONAI_ROOT / "monai", target / "monai-1.5.1.zip"
    )
    shutil.copy2(
        MONAI_ROOT / "monai-1.5.1.dist-info" / "LICENSE",
        target / "MONAI_LICENSE",
    )

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
            "purpose": "Independent multimodal LSM-FM detector training and clean validation; no submission",
            "pretrained_model": {
                "name": "LSM-FM image-text SwinUNETR feature-36 student",
                "source_doi": "10.5281/zenodo.20146516",
                "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
                "stripped_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "license": "CC-BY-4.0",
                "parameters": 36_032_614,
                "count_scope": "student state tensors, including persistent buffers",
                "input_shape": [64, 64, 64],
                "pretraining_objective": "image-text plus masked reconstruction and distillation",
            },
            "implementation": {
                "name": "MONAI",
                "version": "1.5.1",
                "license": "Apache-2.0",
            },
            "archives": {
                "monai-1.5.1.zip": {
                    "mount_directory": "monai-1.5.1",
                    "files": monai_files,
                }
            },
            "candidate": {
                "architecture": "SwinUNETR feature-36 with a Biohub-owned one-channel heatmap head",
                "parameters": 35_072_515,
                "public_predictions_copied": False,
                "public_kaggle_code_copied": False,
                "teachers_are_frozen_pseudo_label_sources_only": True,
                "validation_movies_excluded_from_training": 12,
                "selective_soft_distillation": False,
                "activation_strategy": "serialized weak/strong backpropagation for 16 GB T4",
            },
            "files": files,
        },
    )
    write_json(
        target / "dataset-metadata.json",
        {
            "title": "Biohub LSM-FM Image-Text PU Runtime v1",
            "id": f"indarkarhana/{DATASET_NAME}",
            "licenses": [{"name": "CC-BY-4.0"}],
            "description": "CC-BY-4.0 feature-36 LSM-FM weights with Apache-2.0 MONAI 1.5.1; full attribution and licenses included.",
            "isPrivate": True,
        },
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "files": len(files) + 2,
                "bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
            }
        )
    )


if __name__ == "__main__":
    main()
