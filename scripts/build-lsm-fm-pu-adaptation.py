from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-spatialdino-pu-adaptation.py"
RUN_ID = "lsm-fm-pu-adaptation-v1"
TARGET = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = TARGET / f"biohub-{RUN_ID}.ipynb"


def replace_exact(source: str, old: str, new: str, *, count: int | None = None) -> str:
    observed = source.count(old)
    if observed == 0 or (count is not None and observed != count):
        raise RuntimeError(
            f"base builder drift for {old!r}: expected {count}, observed {observed}"
        )
    return source.replace(old, new)


def main() -> None:
    base = runpy.run_path(str(BASE_BUILDER))
    code_cell = base["code_cell"]
    markdown_cell = base["markdown_cell"]

    watchdog = replace_exact(base["WATCHDOG"], "spatialdino-pu-adaptation-v1", RUN_ID)
    watchdog = replace_exact(
        watchdog,
        'print("SpatialDINO PU watchdog armed for 6,900 seconds.")',
        'print("LSM-FM PU watchdog armed for 6,900 seconds.")',
        count=1,
    )
    watchdog = replace_exact(watchdog, "spatialdino_pu_model", "lsm_fm_pu_model")
    watchdog = replace_exact(
        watchdog,
        "spatialdino_pu_validation.json",
        "lsm_fm_pu_validation.json",
    )
    watchdog = replace_exact(watchdog, "spatialdino_pu_validation", "lsm_fm_pu_validation")

    setup = replace_exact(
        base["SETUP"],
        "biohub-spatialdino-pu-runtime-v1",
        "biohub-lsm-fm-pu-runtime-v1",
    )
    setup = replace_exact(
        setup,
        'checkpoint = runtime / "spatialdino_vits8_backbone.pth"\n'
        'if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8":\n'
        '    raise RuntimeError("SpatialDINO checkpoint hash mismatch")',
        'checkpoint = runtime / "lsm_fm_image_only_student.pt"\n'
        'if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0":\n'
        '    raise RuntimeError("stripped LSM-FM checkpoint hash mismatch")',
        count=1,
    )
    setup = replace_exact(
        setup,
        'for module in ("numpy", "scipy", "torch", "tracksdata", "zarr"):',
        'for module in ("numpy", "scipy", "torch", "monai", "tracksdata", "zarr"):',
        count=1,
    )
    setup = replace_exact(
        setup,
        'sys.path.insert(0, str(support_repo.parent))\n'
        'sys.path.insert(0, str(runtime))',
        'sys.path.insert(0, str(support_repo.parent))\n'
        'manifest_path = runtime / "SOURCE_MANIFEST.json"\n'
        'if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != "fe6077dd0618f4fd2e6b026d88bf18323a66b27b7cebd5bb9431b2e1d3d6735c":\n'
        '    raise RuntimeError("LSM-FM runtime manifest hash mismatch")\n'
        'manifest = json.loads(manifest_path.read_text(encoding="utf-8"))\n'
        'monai_archive = runtime / "monai-1.5.1.zip"\n'
        'monai_mount = runtime / manifest["archives"]["monai-1.5.1.zip"]["mount_directory"]\n'
        'if monai_archive.is_file():\n'
        '    if hashlib.sha256(monai_archive.read_bytes()).hexdigest() != "207d0be1dee0ba5a164ea553bf0f49a316a08267e17c6729b595553aac46117d":\n'
        '        raise RuntimeError("MONAI runtime archive hash mismatch")\n'
        '    monai_import_root = monai_archive\n'
        'elif monai_mount.is_dir():\n'
        '    for name, row in manifest["archives"]["monai-1.5.1.zip"]["files"].items():\n'
        '        path = monai_mount / name\n'
        '        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:\n'
        '            raise RuntimeError(f"extracted MONAI file hash mismatch: {name}")\n'
        '    monai_import_root = monai_mount\n'
        'else:\n'
        '    raise FileNotFoundError("MONAI archive or extracted Kaggle resource is missing")\n'
        'sys.path.insert(0, str(monai_import_root))\n'
        'sys.path.insert(0, str(runtime))',
        count=1,
    )
    setup = replace_exact(
        setup,
        'manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))\n'
        'for name, row in manifest["files"].items():\n'
        '    path = runtime / name\n'
        '    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:\n'
        '        raise RuntimeError(f"runtime file hash mismatch: {name}")',
        'for name, row in manifest["files"].items():\n'
        '    path = runtime / name\n'
        '    if name == "monai-1.5.1.zip" and monai_mount.is_dir():\n'
        '        continue\n'
        '    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:\n'
        '        raise RuntimeError(f"runtime file hash mismatch: {name}")',
        count=1,
    )

    train = replace_exact(base["TRAIN"], "spatialdino_pu_model", "lsm_fm_pu_model")
    train = replace_exact(
        train,
        '    "--spatialdino-checkpoint", str(checkpoint),',
        '    "--model-family", "lsm_fm",\n'
        '    "--lsm-fm-checkpoint", str(checkpoint),\n'
        '    "--lsm-fm-checkpoint-sha256", "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0",',
        count=1,
    )
    train = replace_exact(train, '    "--encoder-blocks", "4",', '    "--encoder-blocks", "2",', count=1)
    train = replace_exact(
        train,
        "Launching independent SpatialDINO PU adaptation:",
        "Launching independent LSM-FM PU adaptation:",
        count=1,
    )

    validate = replace_exact(
        base["VALIDATE"],
        'validation_result = validation_dir / "spatialdino_pu_validation.json"',
        'validation_result = validation_dir / "lsm_fm_pu_validation.json"',
        count=1,
    )
    validate = replace_exact(
        validate, "spatialdino_pu_validation", "lsm_fm_pu_validation"
    )
    validate = replace_exact(
        validate,
        '    "--spatialdino-checkpoint", str(checkpoint),',
        '    "--model-family", "lsm_fm",\n'
        '    "--lsm-fm-checkpoint", str(checkpoint),\n'
        '    "--lsm-fm-checkpoint-sha256", "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0",',
        count=1,
    )
    validate = replace_exact(validate, '    "--batch-size", "4",', '    "--batch-size", "1",', count=1)
    validate = replace_exact(
        validate,
        '    "--max-wall-seconds", "1300",',
        '    "--max-wall-seconds", "1300",\n'
        '    "--output-name", "lsm_fm_pu_validation.json",',
        count=1,
    )
    finish = replace_exact(
        base["FINISH"],
        "SpatialDINO PU training and clean validation complete; no submission was created.",
        "LSM-FM PU training and clean validation complete; no submission was created.",
        count=1,
    )

    TARGET.mkdir(parents=True, exist_ok=True)
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            code_cell(watchdog),
            markdown_cell(
                "# Independent 3D light-sheet SwinUNETR detector\n\n"
                "The openly licensed LSM-FM image-only student initializes a new "
                "one-channel Biohub heatmap head on geometry-matched isotropic 64-cubed "
                "volumes. Public TemporalUNets are frozen PU teachers only; all twelve "
                "validation movies remain excluded and no submission is created.\n"
            ),
            code_cell(setup),
            code_cell(train),
            code_cell(validate),
            code_cell(finish),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/biohub-{RUN_ID}",
        "title": "Biohub LSM-FM PU Adaptation v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "swinunetr", "light-sheet", "positive-unlabeled"],
        "dataset_sources": [
            "indarkarhana/biohub-lsm-fm-pu-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
        ],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
