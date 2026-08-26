from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-spatialdino-pu-adaptation.py"
RUN_ID = "spatialdino-pu-selective-distillation-v2"
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

    watchdog = replace_exact(
        base["WATCHDOG"], "spatialdino-pu-adaptation-v1", RUN_ID
    )
    setup = replace_exact(
        base["SETUP"],
        "biohub-spatialdino-pu-runtime-v1",
        "biohub-spatialdino-pu-runtime-v2",
    )
    train = replace_exact(
        base["TRAIN"],
        '    "--ema-decay", "0.995",\n',
        '    "--ema-decay", "0.995",\n'
        '    "--distillation-weight", "0.25",\n'
        '    "--distillation-support-threshold", "0.05",\n'
        '    "--distillation-agreement-power", "2.0",\n',
        count=1,
    )
    train = replace_exact(
        train,
        "Launching independent SpatialDINO PU adaptation:",
        "Launching independent SpatialDINO selective-distillation adaptation:",
        count=1,
    )
    finish = replace_exact(
        base["FINISH"],
        "SpatialDINO PU training and clean validation complete; no submission was created.",
        "SpatialDINO selective-distillation training and clean validation complete; no submission was created.",
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
                "# Independent SpatialDINO selective-distillation detector\n\n"
                "The same 29.5M-parameter SpatialDINO/UNETR student learns a minority "
                "soft loss only on two-seed-supported voxels. Teacher disagreement is "
                "quadratically discounted, organizer positives remain authoritative, all "
                "twelve validation movies remain excluded, and no submission is created.\n"
            ),
            code_cell(setup),
            code_cell(train),
            code_cell(base["VALIDATE"]),
            code_cell(finish),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/biohub-{RUN_ID}",
        "title": "Biohub SpatialDINO Selective Distillation v2",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [
            "gpu",
            "spatialdino",
            "unetr",
            "positive-unlabeled",
            "distillation",
        ],
        "dataset_sources": [
            "indarkarhana/biohub-spatialdino-pu-runtime-v2",
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
