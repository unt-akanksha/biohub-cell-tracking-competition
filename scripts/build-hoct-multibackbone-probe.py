from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-hoct-probe-finetune.py"
TARGET = ROOT / "kaggle" / "biohub-hoct-multibackbone-probe-v1"
NOTEBOOK = TARGET / "biohub-hoct-multibackbone-probe-v1.ipynb"


def _base() -> dict:
    return runpy.run_path(str(BASE_BUILDER))


def main() -> None:
    source = _base()
    code_cell = source["code_cell"]
    markdown_cell = source["markdown_cell"]

    watchdog = (
        source["WATCHDOG"]
        .replace("hoct-probe-finetune-v1", "hoct-multibackbone-probe-v1")
        .replace("hoct_probe_v1", "hoct_multibackbone_v1")
        .replace("HOCT probe watchdog", "HOCT multibackbone watchdog")
    )
    setup = (
        source["SETUP"]
        .replace("biohub-hoct-runtime-v1", "biohub-hoct-multibackbone-runtime-v1")
        .replace(
            'for name in ("biohub_adapter.py", "train_biohub_hoct_probe.py", "general_v1.pt"):',
            'for name in (\n'
            '    "association_ensemble.py", "biohub_adapter.py", "multibackbone.py",\n'
            '    "train_biohub_hoct_probe.py", "train_biohub_hoct_multibackbone.py",\n'
            '    "general_v1.pt", "ctc_v0.pt",\n'
            '):',
        )
        .replace(
            '"source_model_sha256": hoct_manifest["pretrained_model"]["sha256"],',
            '"source_model_sha256": {name: value["sha256"] for name, value in hoct_manifest["pretrained_models"].items()},',
        )
    )
    train = (
        source["TRAIN"]
        .replace("hoct_probe_v1", "hoct_multibackbone_v1")
        .replace("train_biohub_hoct_probe.py", "train_biohub_hoct_multibackbone.py")
        .replace(
            '"--pretrained-model", str(hoct_runtime / "general_v1.pt"),',
            '"--general-model", str(hoct_runtime / "general_v1.pt"),\n'
            '    "--ctc-model", str(hoct_runtime / "ctc_v0.pt"),',
        )
        .replace("Launching Biohub HOCT probe:", "Launching Biohub HOCT multibackbone probe:")
        .replace(
            '"feature_extraction": training["feature_extraction"],',
            '"training_evidence": training["training_evidence"],',
        )
    )
    finish = source["FINISH"].replace(
        "HOCT probe experiment", "HOCT multibackbone experiment"
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
                "# Biohub HOCT multi-backbone association adaptation\n\n"
                "This private experiment fits independent linear probes on the official "
                "general_v1 and CTC-specialized ctc_v0 HOCT backbones. It selects among "
                "four single-model heads and six support-aware blends using two complete "
                "movies, then reads two disjoint acceptance movies once. It creates no "
                "competition submission and uses no leaderboard feedback.\n"
            ),
            code_cell(setup),
            code_cell(train),
            code_cell(finish),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": "indarkarhana/biohub-hoct-multibackbone-probe-v1",
        "title": "Biohub HOCT Multibackbone Probe v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "transformer", "ensemble"],
        "dataset_sources": [
            "indarkarhana/biohub-hoct-multibackbone-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        ],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
