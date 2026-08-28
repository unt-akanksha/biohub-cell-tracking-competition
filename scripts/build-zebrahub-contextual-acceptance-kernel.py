from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "zebrahub-contextual-acceptance-evaluation-v1"
KERNEL_ID = f"biohub-{RUN_ID}"
REMOTE_KERNEL_ID = "biohub-zsns001-contextual-gate-v1"
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

RUN_ID = "zebrahub-contextual-acceptance-evaluation-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")
OUTPUT_DIR = Path("/kaggle/working/zebrahub_contextual_acceptance_v1")
DECLARED_BUDGET_SECONDS = 3_600


def write_terminal(status, error=None):
    acceptance_terminal = OUTPUT_DIR / "acceptance_terminal.json"
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": DECLARED_BUDGET_SECONDS,
        "evaluator_hard_stop_seconds": 3_300,
        "acceptance_terminal_exists": acceptance_terminal.is_file(),
        "gpu_count_required": 2,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if error is not None:
        payload["error"] = str(error)
    if acceptance_terminal.is_file():
        payload["acceptance_terminal_sha256"] = hashlib.sha256(
            acceptance_terminal.read_bytes()
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
print("ZSNS001 one-shot acceptance watchdog armed for 3,600 seconds.")
'''


SETUP = r'''import subprocess
import sys

import torch

EXPECTED_RUNTIME_MANIFEST_SHA256 = "6aefc98b953a15b853731bc970a9734ddcab08a51938bcc9769f29fa6bba1f29"
EXPECTED_ACCEPTANCE_MANIFEST_SHA256 = "cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0"


def unique_parent(filename, expected_sha256=None, run_id=None):
    matches = []
    for path in Path("/kaggle/input").rglob(filename):
        if expected_sha256 is not None:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected_sha256:
                continue
        if run_id is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("run_id") != run_id:
                continue
        matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError({
            "filename": filename,
            "expected_sha256": expected_sha256,
            "run_id": run_id,
            "matches": [str(path) for path in matches],
        })
    return matches[0]


if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"ZSNS001 acceptance requires exactly two GPUs, saw {torch.cuda.device_count()}"
    )
gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
print({"gpu_count": 2, "gpu_names": gpu_names})

runtime = unique_parent(
    "SOURCE_MANIFEST.json",
    expected_sha256=EXPECTED_RUNTIME_MANIFEST_SHA256,
)
acceptance_root = unique_parent(
    "DATASET_MANIFEST.json",
    expected_sha256=EXPECTED_ACCEPTANCE_MANIFEST_SHA256,
)
pretraining_root = unique_parent(
    "pretraining_terminal.json",
    run_id="zebrahub-contextual-pretrain-v1",
)
runtime_verification = subprocess.run(
    [
        sys.executable,
        str(runtime / "verify_runtime.py"),
        "--root",
        str(runtime),
        "--require-gpus",
    ],
    check=True,
    capture_output=True,
    text=True,
)
runtime_evidence = json.loads(runtime_verification.stdout)
acceptance_verification = subprocess.run(
    [
        sys.executable,
        str(runtime / "verify_zebrahub_contextual_acceptance.py"),
        "--root",
        str(acceptance_root),
    ],
    check=True,
    capture_output=True,
    text=True,
)
acceptance_evidence = json.loads(acceptance_verification.stdout)
if not (
    runtime_evidence.get("status") == "verified"
    and runtime_evidence.get("required_gpu_count") == 2
    and runtime_evidence.get("detected_gpu_count") == 2
    and runtime_evidence.get("submission_command_included") is False
    and acceptance_evidence.get("status") == "verified_unopened"
    and acceptance_evidence.get("manifest_sha256")
        == EXPECTED_ACCEPTANCE_MANIFEST_SHA256
    and acceptance_evidence.get("model_predictions_read") is False
    and acceptance_evidence.get("competition_test_data_read") is False
    and acceptance_evidence.get("public_competition_predictions_read") is False
    and acceptance_evidence.get("leaderboard_used") is False
    and acceptance_evidence.get("submission_created") is False
):
    raise RuntimeError({
        "runtime_evidence": runtime_evidence,
        "acceptance_evidence": acceptance_evidence,
    })
print(json.dumps({
    "runtime": str(runtime),
    "acceptance_root": str(acceptance_root),
    "pretraining_root": str(pretraining_root),
    "runtime_evidence": runtime_evidence,
    "acceptance_evidence": acceptance_evidence,
}, indent=2, sort_keys=True))
'''


EVALUATE = r'''command = [
    sys.executable,
    str(runtime / "evaluate_zebrahub_contextual_acceptance.py"),
    "--orchestrate",
    "--pretraining-root", str(pretraining_root),
    "--acceptance-root", str(acceptance_root),
    "--output-dir", str(OUTPUT_DIR),
    "--patch-batch-size", "48",
    "--hard-stop-seconds", "3300",
]
print("Launching isolated two-GPU ZSNS001 acceptance:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as error:
    write_terminal("failed", error)
    raise

acceptance_terminal = OUTPUT_DIR / "acceptance_terminal.json"
if not acceptance_terminal.is_file():
    raise RuntimeError("Acceptance evaluator exited without terminal evidence")
result = json.loads(acceptance_terminal.read_text(encoding="utf-8"))
folds = result.get("folds", {})
valid_folds = bool(
    set(folds) == {"target_44b6", "target_6bba"}
    and all(
        row.get("status") == "completed"
        and row.get("gate_passed") is True
        and row.get("selection_or_checkpoint_redirect_permitted") is False
        and row.get("competition_data_read") is False
        and row.get("public_predictions_copied") is False
        and row.get("public_leaderboard_used_for_selection") is False
        and row.get("submission_created") is False
        for row in folds.values()
    )
)
if not (
    result.get("status") == "completed"
    and result.get("run_id") == RUN_ID
    and result.get("gpu_count") == 2
    and result.get("both_folds_improved") is True
    and result.get("acceptance_manifest_sha256")
        == EXPECTED_ACCEPTANCE_MANIFEST_SHA256
    and result.get("selection_or_checkpoint_redirect_permitted") is False
    and result.get("competition_data_read") is False
    and result.get("public_predictions_copied") is False
    and result.get("public_leaderboard_used_for_selection") is False
    and result.get("submission_created") is False
    and valid_folds
):
    raise RuntimeError(f"Invalid ZSNS001 acceptance terminal: {result}")
print(json.dumps(result, indent=2, sort_keys=True))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("One-shot external acceptance complete; no competition or submission path was used.")
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
                "# Frozen third-embryo acceptance\n\n"
                "Two independently trained project-authored contextual models are "
                "evaluated once on the untouched ZSNS001 shards, one fold per GPU. "
                "This notebook cannot redirect checkpoint selection, read competition "
                "data, copy public predictions, inspect a leaderboard, or submit.\n"
            ),
            code_cell(SETUP),
            code_cell(EVALUATE),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{REMOTE_KERNEL_ID}",
        "title": "Biohub ZSNS001 Contextual Gate v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "acceptance", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-zebrahub-contextual-acceptance-runtime-v1",
            "indarkarhana/biohub-zebrahub-contextual-acceptance-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-zebrahub-contextual-pretrain-v1"
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
