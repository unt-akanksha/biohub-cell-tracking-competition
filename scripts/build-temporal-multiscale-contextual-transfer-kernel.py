from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-temporal-contextual-transfer-kernel.py"
KERNEL_ID = "biohub-temporal-multiscale-transfer-v4"
TARGET = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = TARGET / f"{KERNEL_ID}.ipynb"
RUN_ID = "temporal-multiscale-contextual-pair-fusion-v4"
RUNTIME_MANIFEST_SHA256 = (
    "ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3"
)


def replace_exact(text: str, old: str, new: str, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"base contextual transfer builder drifted for {old!r}: "
            f"expected {count}, saw {actual}"
        )
    return text.replace(old, new)


MULTISCALE_PRETRAINING_VALIDATION = r'''

# The transfer weights must come from the distinct accepted-v3-warm-started v4 lane.
multiscale_pretraining_root = unique_json_parent(
    "pretraining_terminal.json",
    lambda payload, parent: (
        payload.get("run_id") == "zebrahub-multiscale-contextual-pretrain-v1"
        and payload.get("status") == "completed"
        and payload.get("appearance_family")
            == "temporal_multiscale_contextual_pair_fusion_v4"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and payload.get("competition_data_read") is False
        and payload.get("public_predictions_copied") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
        and (parent / "target_44b6" / "pretrained_model.pt").is_file()
        and (parent / "target_6bba" / "pretrained_model.pt").is_file()
    ),
)
multiscale_aggregate = json.loads(
    (multiscale_pretraining_root / "pretraining_terminal.json").read_text(
        encoding="utf-8"
    )
)
multiscale_folds = multiscale_aggregate.get("folds", {})
if set(multiscale_folds) != set(ACCEPTANCE_FOLDS):
    raise RuntimeError("Multiscale pretraining fold inventory changed")

from multiscale_contextual_pair_fusion import (
    MultiscaleContextualPairFusionAssociationModel,
)

for fold in ACCEPTANCE_FOLDS:
    model_path = multiscale_pretraining_root / fold / "pretrained_model.pt"
    worker_path = multiscale_pretraining_root / fold / "worker_terminal.json"
    if not worker_path.is_file():
        raise FileNotFoundError(f"Multiscale worker evidence is missing: {fold}")
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    initialization = worker.get("initialization", {})
    model_sha256 = sha256_file(model_path)
    aggregate_row = multiscale_folds[fold]
    if not (
        worker.get("status") == "completed"
        and worker.get("run_id")
            == "zebrahub-multiscale-contextual-pretrain-v1"
        and worker.get("fold") == fold
        and worker.get("appearance_family")
            == "temporal_multiscale_contextual_pair_fusion_v4"
        and worker.get("parameter_count") == 46_386_607
        and worker.get("model_sha256") == model_sha256
        and aggregate_row.get("model_sha256") == model_sha256
        and worker.get("selection_gate_passed") is True
        and worker.get("audit_gate_passed") is True
        and int(worker.get("best_step", 0)) > 0
        and initialization.get("run_id") == "zebrahub-contextual-pretrain-v1"
        and initialization.get("source_family")
            == "temporal_contextual_pair_fusion_v3"
        and initialization.get("target_family")
            == "temporal_multiscale_contextual_pair_fusion_v4"
        and initialization.get("initial_predictions_numerically_preserved") is True
        and worker.get("competition_data_read") is False
        and worker.get("public_predictions_copied") is False
        and worker.get("public_leaderboard_used_for_selection") is False
        and worker.get("submission_created") is False
    ):
        raise RuntimeError(f"Multiscale pretraining evidence is invalid: {fold}")
    strict_multiscale = MultiscaleContextualPairFusionAssociationModel()
    strict_multiscale.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), strict=True
    )
    if sum(parameter.numel() for parameter in strict_multiscale.parameters()) != 46_386_607:
        raise RuntimeError(f"Multiscale checkpoint architecture changed: {fold}")
    del strict_multiscale
print("Both v4 pretraining checkpoints strict-loaded for reciprocal transfer.")
'''


def transformed_cells() -> tuple[str, str, str, str]:
    base = runpy.run_path(str(BASE_BUILDER))
    watchdog = replace_exact(
        base["WATCHDOG"], "temporal-contextual-pair-fusion-v3", RUN_ID
    )
    watchdog = replace_exact(
        watchdog,
        "temporal_contextual_transfer_v3",
        "temporal_multiscale_contextual_transfer_v4",
    )
    watchdog = replace_exact(
        watchdog, "Contextual transfer", "Multiscale contextual transfer"
    )

    setup = replace_exact(
        base["SETUP"],
        "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d",
        RUNTIME_MANIFEST_SHA256,
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
    )
    setup = replace_exact(
        setup,
        "Contextual transfer runtime changed",
        "Multiscale contextual runtime changed",
    )
    setup += MULTISCALE_PRETRAINING_VALIDATION

    train = replace_exact(
        base["TRAIN"],
        'str(runtime / "train_dual_fold_contextual_pair_fusion.py")',
        'str(runtime / "train_dual_fold_multiscale_contextual_pair_fusion.py")',
    )
    train = replace_exact(
        train, "str(pretraining_root)", "str(multiscale_pretraining_root)"
    )
    train = replace_exact(
        train,
        "temporal_contextual_pair_fusion_v3",
        "temporal_multiscale_contextual_pair_fusion_v4",
        count=3,
    )
    train = replace_exact(
        train,
        "hash-bound ZebraHub external pretraining",
        "hash-bound multiscale ZebraHub external pretraining",
    )
    train = replace_exact(
        train,
        "Launching two-GPU contextual transfer",
        "Launching two-GPU multiscale contextual transfer",
    )
    train = replace_exact(
        train,
        "Contextual trainer exited without terminal evidence",
        "Multiscale trainer exited without terminal evidence",
    )
    train = replace_exact(
        train,
        "Invalid contextual transfer terminal",
        "Invalid multiscale contextual transfer terminal",
    )
    finish = replace_exact(
        base["FINISH"],
        "Contextual transfer complete",
        "Multiscale contextual transfer complete",
    )
    return watchdog, setup, train, finish


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
    watchdog, setup, train, finish = transformed_cells()
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
                "# Project-authored multiscale contextual transfer v4\n\n"
                "Both v4 ZebraHub folds must bind accepted v3 warm starts and pass "
                "fixed ZSNS005 gates before reciprocal Biohub transfer. The transfer "
                "then requires improvement on both disjoint real folds while retaining "
                "synthetic performance. Public predictions, leaderboard selection, "
                "artifact creation, and submission are unavailable.\n"
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
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Temporal Multiscale Transfer v4",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "multiscale", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "josefreitasalvesneto/biohub-synthetic-dataset",
            "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
            "indarkarhana/biohub-zsns001-contextual-gate-v1",
            "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1",
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
