from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-pretrain-kernel.py"
TRANSFER_BUILDER = ROOT / "scripts" / "build-temporal-contextual-transfer-kernel.py"
KERNEL_ID = "biohub-zebrahub-multiscale-pretrain-v1"
TARGET = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = TARGET / f"{KERNEL_ID}.ipynb"
RUN_ID = "zebrahub-multiscale-contextual-pretrain-v1"
RUNTIME_MANIFEST_SHA256 = (
    "ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3"
)


def replace_exact(text: str, old: str, new: str, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"base contextual builder drifted for {old!r}: expected {count}, saw {actual}"
        )
    return text.replace(old, new)


def between(text: str, start: str, end: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"base transfer builder markers drifted: {start!r}, {end!r}")
    return text[text.index(start) : text.index(end)]


def accepted_v3_setup(transfer_setup: str) -> str:
    helpers = between(
        transfer_setup,
        "def unique_json_parent(filename, predicate):",
        "if torch.cuda.device_count() != 2:",
    )
    discovery = between(
        transfer_setup,
        "pretraining_root = unique_json_parent(",
        "if None in (runtime_input, competition, support, synthetic):",
    )
    validation = between(
        transfer_setup,
        "pretraining_terminal_path = pretraining_root / \"pretraining_terminal.json\"",
        "sys.path.insert(0, str(runtime))",
    )
    return f'''\n\n# Bind the warm start to the independently accepted v3 output.\nimport math\n\nEXPECTED_ACCEPTANCE_MANIFEST_SHA256 = "cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0"\nEXPECTED_ACCEPTANCE_INVENTORY_SHA256 = "e32bc686e14222e43acb8d6247351e286eae8ed6fdb1f4ab5087e55fb0c79667"\nEXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256 = "05ad8b3195aa2786ffb8a2ffcb247d118d1e5cf96026264e0534c468979e6e0e"\nACCEPTANCE_FOLDS = ("target_44b6", "target_6bba")\nACCEPTANCE_SEEDS = {{"target_44b6": 51004, "target_6bba": 61007}}\n\n{helpers}\n{discovery}\n{validation}\nsys.path.insert(0, str(runtime))\nfrom contextual_pair_fusion import ContextualPairFusionAssociationModel\n\nfor fold in ACCEPTANCE_FOLDS:\n    strict_probe = ContextualPairFusionAssociationModel()\n    strict_probe.load_state_dict(\n        torch.load(\n            pretraining_root / fold / "pretrained_model.pt",\n            map_location="cpu",\n            weights_only=True,\n        ),\n        strict=True,\n    )\n    if sum(parameter.numel() for parameter in strict_probe.parameters()) != 20_747_761:\n        raise RuntimeError(f"Accepted contextual checkpoint architecture changed: {{fold}}")\n    del strict_probe\nprint("Both accepted v3 checkpoints strict-loaded for multiscale warm start.")\n'''


def transformed_cells() -> tuple[str, str, str, str]:
    base = runpy.run_path(str(BASE_BUILDER))
    transfer = runpy.run_path(str(TRANSFER_BUILDER))

    watchdog = replace_exact(
        base["WATCHDOG"],
        "zebrahub-contextual-pretrain-v1-mp-repair",
        RUN_ID,
    )
    watchdog = replace_exact(
        watchdog, "zebrahub-contextual-pretrain-v1", RUN_ID
    )
    watchdog = replace_exact(
        watchdog,
        "zebrahub_contextual_pretrain_v1",
        "zebrahub_multiscale_contextual_pretrain_v1",
    )

    setup = replace_exact(
        base["SETUP"],
        "aff4e21f675848f94cbfd42e4d1a43ebc5b470fbc21db3934f85797066fc65cb",
        RUNTIME_MANIFEST_SHA256,
    )
    setup = replace_exact(
        setup,
        "biohub-temporal-contextual-pair-fusion-runtime-v3",
        "biohub-temporal-multiscale-contextual-runtime-v4",
        count=2,
    )
    setup = replace_exact(
        setup,
        "contextual_pair_fusion_runtime_v3",
        "multiscale_contextual_runtime_v4",
    )
    setup += accepted_v3_setup(transfer["SETUP"])

    train = replace_exact(
        base["TRAIN"],
        'str(runtime / "train_zebrahub_contextual_pretrain.py")',
        'str(runtime / "train_zebrahub_multiscale_contextual_pretrain.py")',
    )
    train = replace_exact(
        train,
        '"--output-dir", str(OUTPUT_DIR),',
        '"--output-dir", str(OUTPUT_DIR),\n    "--initial-model-root", str(pretraining_root),',
    )
    train = replace_exact(
        train, '"--patch-batch-size", "48"', '"--patch-batch-size", "32"'
    )
    train = replace_exact(
        train, '"--gradient-accumulation", "2"', '"--gradient-accumulation", "3"'
    )
    train = replace_exact(
        train,
        'print("Launching two-fold ZebraHub contextual pretraining:", " ".join(command))',
        'print("Launching two-fold ZebraHub multiscale contextual pretraining:", " ".join(command))',
    )
    train = replace_exact(
        train,
        'row.get("status") == "completed"',
        'row.get("status") == "completed"\n'
        '        and row.get("appearance_family")\n'
        '            == "temporal_multiscale_contextual_pair_fusion_v4"\n'
        '        and row.get("parameter_count") == 46_386_607\n'
        '        and row.get("initialization", {}).get("run_id")\n'
        '            == "zebrahub-contextual-pretrain-v1"\n'
        '        and row.get("initialization", {}).get("source_family")\n'
        '            == "temporal_contextual_pair_fusion_v3"\n'
        '        and row.get("initialization", {}).get("target_family")\n'
        '            == "temporal_multiscale_contextual_pair_fusion_v4"\n'
        '        and row.get("initialization", {}).get(\n'
        '            "initial_predictions_numerically_preserved"\n'
        '        ) is True',
    )
    train = replace_exact(
        train,
        'and result.get("gpu_count") == 2',
        'and result.get("appearance_family")\n'
        '        == "temporal_multiscale_contextual_pair_fusion_v4"\n'
        '    and result.get("gpu_count") == 2',
    )
    finish = replace_exact(
        base["FINISH"],
        "External pretraining complete",
        "Multiscale external pretraining complete",
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
                "# Project-authored multiscale contextual pretraining v1\n\n"
                "The 46.4M-parameter residual expansion starts only from both strictly "
                "loaded v3 checkpoints bound to passing one-shot ZSNS001 acceptance. "
                "Two isolated GPUs train on ZSNS004 and select only on fixed ZSNS005 "
                "shards. No competition data, public predictions, leaderboard "
                "selection, or submission path is available.\n"
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
        "title": "Biohub ZebraHub Multiscale Pretrain v1",
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
            "indarkarhana/biohub-zebrahub-contextual-shards-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
            "indarkarhana/biohub-zsns001-contextual-gate-v1",
        ],
        "competition_sources": [],
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
