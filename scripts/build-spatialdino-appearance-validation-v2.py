from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-spatialdino-appearance-validation.py"
TARGET = ROOT / "kaggle" / "biohub-spatialdino-appearance-validation-v2"
NOTEBOOK = TARGET / "biohub-spatialdino-appearance-validation-v2.ipynb"


def load_base_builder():
    spec = importlib.util.spec_from_file_location("spatialdino_appearance_v1_builder", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected one builder fragment, found {source.count(old)}")
    return source.replace(old, new)


def main() -> None:
    base = load_base_builder()
    watchdog = replace_once(
        base.WATCHDOG,
        'RUN_ID = "spatialdino-appearance-validation-v1"',
        'RUN_ID = "spatialdino-appearance-validation-v2"',
    ).replace("spatialdino_appearance_v1", "spatialdino_appearance_v2")
    old_input = '''hoct_output = first_existing([
    Path("/kaggle/input/biohub-hoct-multibackbone-probe-v1"),
    Path("/kaggle/input/kernels/indarkarhana/biohub-hoct-multibackbone-probe-v1"),
])'''
    new_input = '''hoct_output = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-hoct-processed-validation-v1"),
    Path("/kaggle/input/biohub-hoct-processed-validation-v1"),
])'''
    setup = replace_once(base.SETUP, old_input, new_input)
    # The materialized dataset must itself prove the exact producer artifacts.
    setup = replace_once(
        setup,
        'valid_topologies = []\nfor launcher_path in hoct_output.rglob("launcher_terminal.json"):',
        '''topology_manifest = json.loads((hoct_output / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name, row in topology_manifest["files"].items():
    if sha256_file(hoct_output / name) != row["sha256"]:
        raise RuntimeError(f"materialized topology hash mismatch: {name}")
if topology_manifest.get("ground_truth_labels_in_processed_csv") is not False:
    raise RuntimeError("processed topology provenance does not exclude ground-truth labels")

valid_topologies = []
for launcher_path in hoct_output.rglob("launcher_terminal.json"):''',
    )
    run = base.RUN.replace("spatialdino_appearance_v1", "spatialdino_appearance_v2")

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
            base.code_cell(watchdog),
            base.markdown_cell(
                """
                # SpatialDINO degree-preserving appearance correction v2

                This environment-only retry consumes the completed HOCT output
                through a private, hash-bound dataset because Kaggle did not
                materialize the v1 kernel-output attachment. Scientific code,
                configuration grid, clean split, and acceptance gates are
                unchanged. No submission is created.
                """
            ),
            base.code_cell(setup),
            base.code_cell(run),
            base.code_cell(base.FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": "indarkarhana/biohub-spatialdino-appearance-validation-v2",
        "title": "Biohub SpatialDINO Appearance Validation v2",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "spatialdino", "appearance"],
        "dataset_sources": [
            "indarkarhana/biohub-spatialdino-runtime-v1",
            "indarkarhana/biohub-hoct-multibackbone-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "indarkarhana/biohub-hoct-processed-validation-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
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
