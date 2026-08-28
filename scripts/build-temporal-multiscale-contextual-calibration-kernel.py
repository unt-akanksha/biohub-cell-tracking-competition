from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-temporal-contextual-calibration-kernel.py"
KERNEL_ID = "biohub-multiscale-calibration-v4"
TARGET = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = TARGET / f"{KERNEL_ID}.ipynb"
RUN_ID = "temporal-multiscale-contextual-pair-fusion-blend-v4"
RUNTIME_MANIFEST_SHA256 = (
    "ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3"
)


def replace_exact(text: str, old: str, new: str, *, count: int) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"base contextual calibration builder drifted for {old!r}: "
            f"expected {count}, saw {actual}"
        )
    return text.replace(old, new)


def transformed_cells() -> tuple[str, str, str, str]:
    base = runpy.run_path(str(BASE_BUILDER))
    watchdog = replace_exact(
        base["WATCHDOG"],
        "temporal-contextual-pair-fusion-blend-v3",
        RUN_ID,
        count=1,
    )
    watchdog = replace_exact(
        watchdog,
        "temporal_contextual_calibration_v3",
        "temporal_multiscale_contextual_calibration_v4",
        count=1,
    )
    watchdog = watchdog.replace("Contextual calibration", "Multiscale contextual calibration")

    setup = replace_exact(
        base["SETUP"],
        "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d",
        RUNTIME_MANIFEST_SHA256,
        count=1,
    )
    setup = replace_exact(
        setup,
        "biohub-temporal-contextual-transfer-runtime-v1",
        "biohub-temporal-multiscale-contextual-runtime-v4",
        count=2,
    )
    setup = replace_exact(
        setup,
        "temporal_contextual_transfer_runtime_v1",
        "temporal_multiscale_contextual_runtime_v4",
        count=1,
    )
    setup = replace_exact(
        setup,
        "temporal-contextual-pair-fusion-v3",
        "temporal-multiscale-contextual-pair-fusion-v4",
        count=1,
    )
    setup = replace_exact(
        setup,
        "temporal_contextual_pair_fusion_v3",
        "temporal_multiscale_contextual_pair_fusion_v4",
        count=2,
    )
    setup = setup.replace("Contextual calibration", "Multiscale contextual calibration")

    calibrate = replace_exact(
        base["CALIBRATE"],
        "temporal_contextual_pair_fusion_v3",
        "temporal_multiscale_contextual_pair_fusion_v4",
        count=2,
    )
    calibrate = calibrate.replace(
        "contextual calibration", "multiscale contextual calibration"
    )
    finish = base["FINISH"].replace(
        "Contextual calibration", "Multiscale contextual calibration"
    )
    return watchdog, setup, calibrate, finish


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    watchdog, setup, calibrate, finish = transformed_cells()
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
                "# Multiscale contextual v4 reciprocal calibration\n\n"
                "Exactly two GPUs compare frozen v4 appearance/division evidence "
                "with the pure Trackastra zero-weight control on the twelve reserved "
                "movies per embryo. Processed acceptance, leaderboard selection, "
                "artifact construction, and submission remain excluded.\n"
            ),
            code_cell(setup),
            code_cell(calibrate),
            code_cell(finish),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Temporal Multiscale Contextual Calibration v4",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "multiscale", "calibration"],
        "dataset_sources": [
            "indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-temporal-multiscale-transfer-v4",
            "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
