#!/usr/bin/env python
"""Build the offline exactly-two-GPU division-localization training kernel."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "biohub-multiscale-division-localization-v1"
TARGET = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = TARGET / f"{KERNEL_ID}.ipynb"
RUNTIME_MANIFEST_SHA256 = "528354993b3b64729158d2d64d256243797bb58785b304af42cee908cf0b0e90"
TRAIN_DATASET_MANIFEST_SHA256 = "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
LOCALIZATION_DATASET_MANIFEST_SHA256 = "e1f6eb8c6148c092f72e5eddc81d75f17b16b6f37ad61e35a9d6e652bfb81336"


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


WATCHDOG = r'''
import os
import signal
import threading
import time

HARD_STOP_SECONDS = 23_400

def _watchdog():
    time.sleep(HARD_STOP_SECONDS)
    os.kill(os.getpid(), signal.SIGTERM)

threading.Thread(target=_watchdog, daemon=True).start()
print(f"Notebook watchdog armed for {HARD_STOP_SECONDS} seconds.")
'''


SETUP = rf'''
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def unique_parent(name: str, expected_hash: str, predicate):
    matches = []
    for path in INPUT.rglob(name):
        if path.is_file() and sha256_file(path) == expected_hash:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if predicate(payload):
                matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one verified {{name}}, saw {{matches}}")
    return matches[0]

gpu_lines = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip().splitlines()
if len(gpu_lines) != 2 or any("T4" not in line for line in gpu_lines):
    raise RuntimeError(f"Exactly two T4 GPUs are required, saw {{gpu_lines}}")

runtime = unique_parent(
    "RUNTIME_MANIFEST.json",
    "{RUNTIME_MANIFEST_SHA256}",
    lambda p: (
        p.get("run_id") == "division-localization-runtime-v1"
        and p.get("required_visible_gpu_count") == 2
        and p.get("internet_required") is False
        and p.get("submission_command_included") is False
    ),
)
train_dataset = unique_parent(
    "DATASET_MANIFEST.json",
    "{TRAIN_DATASET_MANIFEST_SHA256}",
    lambda p: (
        p.get("run_id") == "zebrahub-contextual-shards-v1"
        and p.get("competition_test_data_read") is False
        and p.get("leaderboard_used") is False
        and len(p.get("training", {{}}).get("records", [])) == 64
    ),
)
localization_dataset = unique_parent(
    "DATASET_MANIFEST.json",
    "{LOCALIZATION_DATASET_MANIFEST_SHA256}",
    lambda p: (
        p.get("run_id") == "zebrahub-division-localization-shards-v1"
        and p.get("selection_and_audit_frozen_before_extraction") is True
        and p.get("competition_test_data_read") is False
        and p.get("leaderboard_used") is False
    ),
)

v4_matches = []
for path in INPUT.rglob("pretraining_terminal.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if (
        payload.get("run_id") == "zebrahub-multiscale-contextual-pretrain-v1"
        and payload.get("status") == "completed"
        and payload.get("appearance_family")
            == "temporal_multiscale_contextual_pair_fusion_v4"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and payload.get("competition_data_read") is False
        and payload.get("submission_created") is False
    ):
        v4_matches.append(path.parent)
if len(v4_matches) != 1:
    raise RuntimeError(f"Expected one accepted external v4 pretraining output, saw {{v4_matches}}")
v4_root = v4_matches[0]

def materialize_split(source: Path, name: str, target: Path) -> None:
    archive = source / f"{{name}}.zip"
    directory = source / name
    if archive.is_file():
        shutil.unpack_archive(str(archive), str(target))
    elif directory.is_dir():
        shutil.copytree(directory, target, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Missing {{name}} split below {{source}}")

train_root = WORKING / "division_localization_data" / "train"
localization_root = WORKING / "division_localization_data" / "validation"
materialize_split(train_dataset, "train", train_root)
materialize_split(localization_dataset, "selection", localization_root / "selection")
# Extraction does not parse arrays; trainer code cannot address this directory
# until a selection checkpoint has been serialized and hash-frozen.
materialize_split(localization_dataset, "audit", localization_root / "audit")

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
print(json.dumps({{
    "gpu_lines": gpu_lines,
    "runtime": str(runtime),
    "train_root": str(train_root),
    "localization_root": str(localization_root),
    "v4_root": str(v4_root),
}}, indent=2))
'''


TRAIN = r'''
output_root = WORKING / "division_localization"
command = [
    sys.executable,
    str(runtime / "train_dual_fold_division_localization.py"),
    "--orchestrate",
    "--train-root", str(train_root),
    "--localization-root", str(localization_root),
    "--v4-root", str(v4_root),
    "--output-dir", str(output_root),
    "--steps", "6000",
    "--batch-size", "64",
    "--validation-batch-size", "48",
    "--validation-every", "500",
    "--learning-rate", "0.00005",
    "--minimum-learning-rate", "0.0000005",
    "--max-wall-seconds", "21600",
    "--orchestrator-hard-stop-seconds", "22800",
    "--finalization-reserve-seconds", "1200",
]
print("Launching exactly-two-GPU multiscale division localization.", flush=True)
completed = subprocess.run(command, cwd=runtime, check=False)
for log_path in sorted(output_root.glob("*.log")):
    print(f"===== {log_path.name} =====")
    print(log_path.read_text(encoding="utf-8", errors="replace")[-12000:])
if completed.returncode != 0:
    raise RuntimeError(f"Localization trainer failed with code {completed.returncode}")

sys.path.insert(0, str(runtime))
from verify_division_localization_training_output import verify_output

verification = verify_output(output_root, strict_checkpoint=True)
(output_root / "verification.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(verification, indent=2, sort_keys=True))
'''


FINISH = r'''
print("Division localization training and one-shot audit completed.")
print("No competition data was attached and no submission path exists in this kernel.")
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
                "# Project-authored multiscale division localization v1\n\n"
                "This no-submission kernel trains two 47.96M-parameter physical "
                "recentring models from accepted v4 folds. Optimization uses ZSNS004; "
                "new frozen ZSNS005 windows provide checkpoint selection and a sealed "
                "one-shot audit. No competition input, public prediction, leaderboard "
                "choice, or public implementation is available.\n"
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
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Multiscale Division Localization v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "division", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-division-localization-runtime-v1",
            "indarkarhana/biohub-zebrahub-contextual-shards-v1",
            "indarkarhana/biohub-division-localization-shards-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1"
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
