from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "zebrahub-contextual-pretrain-v1"
KERNEL_ID = f"biohub-{RUN_ID}"
TARGET = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = TARGET / f"{KERNEL_ID}.ipynb"


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


WATCHDOG = r'''import atexit
import hashlib
import json
import os
import threading
import time
from pathlib import Path

RUN_ID = "zebrahub-contextual-pretrain-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")
OUTPUT_DIR = Path("/kaggle/working/zebrahub_contextual_pretrain_v1")
DECLARED_BUDGET_SECONDS = 24_000


def write_terminal(status, error=None):
    pretraining_terminal = OUTPUT_DIR / "pretraining_terminal.json"
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": DECLARED_BUDGET_SECONDS,
        "trainer_max_wall_seconds": 21_600,
        "trainer_hard_stop_seconds": 22_800,
        "pretraining_terminal_exists": pretraining_terminal.is_file(),
        "competition_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if error is not None:
        payload["error"] = str(error)
    if pretraining_terminal.is_file():
        payload["pretraining_terminal_sha256"] = hashlib.sha256(
            pretraining_terminal.read_bytes()
        ).hexdigest()
    temporary = TERMINAL.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(TERMINAL)


def abort_terminal():
    if not FINISHED:
        write_terminal("aborted")


def budget_expired():
    write_terminal("budget_expired")
    os._exit(124)


atexit.register(abort_terminal)
TIMER = threading.Timer(DECLARED_BUDGET_SECONDS, budget_expired)
TIMER.daemon = True
TIMER.start()
print("ZebraHub contextual-pretraining watchdog armed for 24,000 seconds.")
'''


SETUP = r'''import shutil
import subprocess
import sys
import zipfile

import torch

EXPECTED_DATASET_MANIFEST_SHA256 = "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
EXPECTED_RUNTIME_MANIFEST_SHA256 = "e7025eb3c6fbb8b648e540f20a5dd4da9aa0982f478efe8bbd999428db04099e"


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


def materialize_runtime_input(runtime_input, runtime):
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    for source in runtime_input.iterdir():
        if source.is_dir():
            shutil.copytree(source, runtime / source.name)
        elif source.is_file() and source.suffix != ".zip" and source.name != "dataset-metadata.json":
            shutil.copy2(source, runtime / source.name)
    for archive in runtime_input.glob("*.zip"):
        destination = runtime / archive.stem
        if destination.exists():
            raise RuntimeError(f"Ambiguous runtime directory and archive: {archive.name}")
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)


def materialize_shard_input(shard_input, shard_root):
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)
    for source in shard_input.iterdir():
        if source.is_dir():
            shutil.copytree(source, shard_root / source.name)
        elif source.is_file() and source.suffix != ".zip" and source.name != "dataset-metadata.json":
            shutil.copy2(source, shard_root / source.name)
    for archive in sorted(shard_input.glob("*.zip")):
        with zipfile.ZipFile(archive) as handle:
            members = [name for name in handle.namelist() if name and not name.endswith("/")]
            roots = {Path(name).parts[0] for name in members}
            if archive.stem in {"train", "validation"} and roots != {archive.stem}:
                destination = shard_root / archive.stem
                if destination.exists():
                    raise RuntimeError(f"Ambiguous shard directory and archive: {archive.name}")
                destination.mkdir(parents=True)
            else:
                destination = shard_root
                if any((shard_root / root).exists() for root in roots):
                    raise RuntimeError(f"Ambiguous shard content and archive: {archive.name}")
            handle.extractall(destination)


if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"ZebraHub contextual pretraining requires exactly two GPUs, saw {torch.cuda.device_count()}"
    )
gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
print({"gpu_count": 2, "gpu_names": gpu_names})

input_root = Path("/kaggle/input")
runtime_input = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-temporal-contextual-pair-fusion-runtime-v3"),
    Path("/kaggle/input/biohub-temporal-contextual-pair-fusion-runtime-v3"),
])
shard_input = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-zebrahub-contextual-shards-v1"),
    Path("/kaggle/input/biohub-zebrahub-contextual-shards-v1"),
])
if runtime_input is None:
    runtime_input = next(
        (
            path.parent
            for path in input_root.rglob("SOURCE_MANIFEST.json")
            if json.loads(path.read_text(encoding="utf-8")).get("run_id")
            == "temporal-contextual-pair-fusion-v3"
        ),
        None,
    )
if shard_input is None:
    shard_input = next(
        (
            path.parent
            for path in input_root.rglob("DATASET_MANIFEST.json")
            if json.loads(path.read_text(encoding="utf-8")).get("run_id")
            == "zebrahub-contextual-shards-v1"
        ),
        None,
    )
if runtime_input is None or shard_input is None:
    raise FileNotFoundError({"runtime_input": runtime_input, "shard_input": shard_input})

runtime = Path("/kaggle/working/contextual_pair_fusion_runtime_v3")
shard_root = Path("/kaggle/working/zebrahub_contextual_shards_v1")
materialize_runtime_input(runtime_input, runtime)
materialize_shard_input(shard_input, shard_root)

runtime_manifest_sha256 = hashlib.sha256(
    (runtime / "SOURCE_MANIFEST.json").read_bytes()
).hexdigest()
if runtime_manifest_sha256 != EXPECTED_RUNTIME_MANIFEST_SHA256:
    raise RuntimeError(f"Runtime manifest changed: {runtime_manifest_sha256}")
subprocess.run(
    [
        sys.executable,
        str(runtime / "verify_runtime.py"),
        "--root",
        str(runtime),
        "--require-gpus",
    ],
    check=True,
)
verification = subprocess.run(
    [
        sys.executable,
        str(runtime / "verify_zebrahub_contextual_dataset.py"),
        "--root",
        str(shard_root),
    ],
    check=True,
    capture_output=True,
    text=True,
)
dataset_evidence = json.loads(verification.stdout)
if not (
    dataset_evidence.get("status") == "verified"
    and dataset_evidence.get("run_id") == "zebrahub-contextual-shards-v1"
    and dataset_evidence.get("manifest_sha256") == EXPECTED_DATASET_MANIFEST_SHA256
    and dataset_evidence.get("training", {}).get("shards") == 64
    and dataset_evidence.get("validation", {}).get("shards") == 16
    and dataset_evidence.get("competition_test_data_read") is False
    and dataset_evidence.get("public_competition_predictions_read") is False
    and dataset_evidence.get("leaderboard_used") is False
    and dataset_evidence.get("submission_created") is False
):
    raise RuntimeError(f"Invalid derived dataset evidence: {dataset_evidence}")

print(json.dumps({
    "runtime": str(runtime),
    "runtime_manifest_sha256": runtime_manifest_sha256,
    "shard_root": str(shard_root),
    "dataset_evidence": dataset_evidence,
}, indent=2, sort_keys=True))
'''


TRAIN = r'''command = [
    sys.executable,
    str(runtime / "train_zebrahub_contextual_pretrain.py"),
    "--orchestrate",
    "--data-root", str(shard_root),
    "--output-dir", str(OUTPUT_DIR),
    "--seed", "51004",
    "--steps", "12000",
    "--minimum-train-shards", "64",
    "--minimum-validation-shards", "16",
    "--patch-batch-size", "48",
    "--gradient-accumulation", "2",
    "--validation-every", "500",
    "--learning-rate", "0.0002",
    "--minimum-learning-rate", "0.000002",
    "--weight-decay", "0.00001",
    "--ema-decay", "0.997",
    "--augmentation-mode", "microscopy_v1",
    "--max-wall-seconds", "21600",
    "--orchestrator-hard-stop-seconds", "22800",
    "--finalization-reserve-seconds", "1200",
]
print("Launching two-fold ZebraHub contextual pretraining:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as error:
    write_terminal("failed", error)
    raise

pretraining_terminal = OUTPUT_DIR / "pretraining_terminal.json"
if not pretraining_terminal.is_file():
    raise RuntimeError("Pretrainer exited without terminal evidence")
result = json.loads(pretraining_terminal.read_text(encoding="utf-8"))
folds = result.get("folds", {})
valid_folds = bool(
    set(folds) == {"target_44b6", "target_6bba"}
    and all(
        row.get("status") == "completed"
        and row.get("dataset_manifest_sha256") == EXPECTED_DATASET_MANIFEST_SHA256
        and row.get("augmentation_mode") == "microscopy_v1"
        and row.get("validation_augmentation") == "none"
        and row.get("reciprocal_parent_loss_weight") == 0.35
        and row.get("shard_cache_policy")
            == "verified immutable tensors preloaded once per GPU"
        and row.get("validation_precision") == "CUDA float16 autocast"
        and row.get("competition_data_read") is False
        and row.get("public_predictions_copied") is False
        and row.get("public_leaderboard_used_for_selection") is False
        and row.get("submission_created") is False
        and (OUTPUT_DIR / fold / "pretrained_model.pt").is_file()
        and hashlib.sha256((OUTPUT_DIR / fold / "pretrained_model.pt").read_bytes()).hexdigest()
            == row.get("model_sha256")
        for fold, row in folds.items()
    )
)
if not (
    result.get("status") == "completed"
    and result.get("run_id") == RUN_ID
    and result.get("gpu_count") == 2
    and isinstance(result.get("both_folds_improved"), bool)
    and result.get("augmentation_mode") == "microscopy_v1"
    and result.get("validation_augmentation") == "none"
    and result.get("competition_data_read") is False
    and result.get("public_predictions_copied") is False
    and result.get("public_leaderboard_used_for_selection") is False
    and result.get("submission_created") is False
    and valid_folds
):
    raise RuntimeError(f"Invalid pretraining terminal: {result}")
print(json.dumps(result, indent=2, sort_keys=True))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("External pretraining complete; no competition data or submission path was used.")
'''


def main() -> None:
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
            code_cell(WATCHDOG),
            markdown_cell(
                "# Original ZebraHub contextual pretraining\n\n"
                "Two independently seeded project-authored 20.7M-parameter contextual "
                "models optimize only the frozen ZSNS004 derived shards and select EMA "
                "checkpoints only on disjoint ZSNS005 shards. Outgoing child and incoming "
                "parent ranking are trained jointly. The competition dataset, "
                "public predictions, leaderboard selection, and submission are excluded.\n"
            ),
            code_cell(SETUP),
            code_cell(TRAIN),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
        "title": "Biohub ZebraHub Contextual Pretrain v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "contextual", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-temporal-contextual-pair-fusion-runtime-v3",
            "indarkarhana/biohub-zebrahub-contextual-shards-v1",
        ],
        "kernel_sources": [],
        "competition_sources": [],
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
