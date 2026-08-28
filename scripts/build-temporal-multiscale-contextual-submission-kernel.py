from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = (
    ROOT / "scripts" / "build-temporal-contextual-submission-candidate-kernel.py"
)
KERNEL_ID = "biohub-multiscale-submission-candidate-v4"
TARGET = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = TARGET / f"{KERNEL_ID}.ipynb"
RUN_ID = "temporal-multiscale-contextual-pair-fusion-candidate-v4"
RUNTIME_MANIFEST_SHA256 = (
    "221ab4359e84706a306cd88b41e2790ebd1a4bad71d2a36ef4a1873499a570b4"
)


def replace_exact(text: str, old: str, new: str, *, count: int) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"base contextual candidate builder drifted for {old!r}: "
            f"expected {count}, saw {actual}"
        )
    return text.replace(old, new)


def transformed_cells() -> tuple[str, str, str, str]:
    base = runpy.run_path(str(BASE_BUILDER))
    watchdog = replace_exact(
        base["WATCHDOG"],
        "temporal-contextual-pair-fusion-candidate-v3",
        RUN_ID,
        count=1,
    )
    watchdog = replace_exact(
        watchdog,
        "temporal_contextual_candidate_v3",
        "temporal_multiscale_contextual_candidate_v4",
        count=1,
    )
    watchdog = watchdog.replace("Contextual submission", "Multiscale submission")

    setup = replace_exact(
        base["SETUP"],
        "e69a20f10f56108818a6bf0715fe071e868fd176d1720cc2ffc04f2a645b41ff",
        RUNTIME_MANIFEST_SHA256,
        count=1,
    )
    setup = replace_exact(
        setup,
        "biohub-temporal-contextual-final-runtime-v1",
        "biohub-multiscale-contextual-final-v4",
        count=2,
    )
    setup = replace_exact(
        setup,
        "temporal_contextual_final_runtime_v1",
        "temporal_multiscale_contextual_final_v4",
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
        count=3,
    )
    setup = replace_exact(
        setup,
        "trackastra_contextual_pair_fusion_blend",
        "trackastra_multiscale_contextual_pair_fusion_blend",
        count=1,
    )
    setup = setup.replace("Contextual candidate", "Multiscale contextual candidate")

    infer = replace_exact(
        base["INFER"],
        "temporal_contextual_pair_fusion_v3",
        "temporal_multiscale_contextual_pair_fusion_v4",
        count=1,
    )
    infer = replace_exact(
        infer,
        "trackastra_contextual_pair_fusion_blend",
        "trackastra_multiscale_contextual_pair_fusion_blend",
        count=1,
    )
    infer = infer.replace("contextual candidate", "multiscale contextual candidate")
    infer = infer.replace("Contextual inference", "Multiscale contextual inference")
    finish = base["FINISH"].replace(
        "Accepted contextual candidate", "Accepted multiscale contextual candidate"
    )
    return watchdog, setup, infer, finish


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
    watchdog, setup, infer, finish = transformed_cells()
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
                "# Accepted multiscale contextual v4 submission candidate\n\n"
                "Exactly two T4 GPUs balance consecutive-frame transition work for "
                "the independently trained 46.4M-parameter v4 folds. Hash-bound exact "
                "processed acceptance is mandatory. The notebook reserves two hours "
                "for finalization, refuses an edge-identical public replica, writes a "
                "local CSV, and contains no upload command.\n"
            ),
            code_cell(setup),
            code_cell(infer),
            code_cell(finish),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Multiscale Submission Candidate v4",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "multiscale", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-multiscale-contextual-final-v4",
            "indarkarhana/biohub-multiscale-exact-acceptance-v4",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-clean-0-927-reproduction-v1",
            "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
            "indarkarhana/biohub-temporal-multiscale-transfer-v4",
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
