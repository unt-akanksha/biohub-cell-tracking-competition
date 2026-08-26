from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-trackastra-graph-finetune-v2"
NOTEBOOK = TARGET / "biohub-trackastra-graph-finetune-v2.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


WATCHDOG = r'''import atexit
import hashlib
import json
import os
import threading
import time
from pathlib import Path

RUN_ID = "trackastra-graph-finetune-v2"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    training_terminal = Path("/kaggle/working/trackastra_graph_v2/training_terminal.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 14400,
        "hard_stop_seconds": 14100,
        "training_terminal_exists": training_terminal.is_file(),
    }
    if error:
        payload["error"] = str(error)
    if training_terminal.is_file():
        payload["training_terminal_sha256"] = hashlib.sha256(training_terminal.read_bytes()).hexdigest()
    temporary = TERMINAL.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(TERMINAL)


def abort_terminal():
    if not FINISHED:
        write_terminal("aborted")


def budget_expired():
    write_terminal("budget_expired")
    os._exit(124)


atexit.register(abort_terminal)
TIMER = threading.Timer(14100, budget_expired)
TIMER.daemon = True
TIMER.start()
print("Trackastra fine-tune watchdog armed for 14,100 seconds.")
'''


SETUP = r'''import importlib
import shutil
import subprocess
import sys
import zipfile


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


input_root = Path("/kaggle/input")
runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-trackastra-graph-runtime-v1"),
    Path("/kaggle/input/biohub-trackastra-graph-runtime-v1"),
])
competition = first_existing([
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
])
support = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
])
if runtime is None:
    runtime = next((p for p in input_root.iterdir() if (p / "SOURCE_MANIFEST.json").is_file()), None)
if support is None:
    support = next((p for p in input_root.iterdir() if (p / "wheels").is_dir()), None)
if runtime is None or competition is None or support is None:
    raise FileNotFoundError({"runtime": runtime, "competition": competition, "support": support})


def materialize_runtime_directory(name, marker):
    direct = runtime / name
    if (direct / marker).exists():
        return direct
    archive = runtime / f"{name}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing runtime directory/archive: {name}")
    destination = Path("/kaggle/working/runtime_materialized") / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    for candidate in (destination, destination / name):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Archive {archive} did not contain {marker}")


ctc_dir = materialize_runtime_directory("ctc", "model.pt")
trackastra_dir = materialize_runtime_directory(
    "trackastra_source", "trackastra/model/model.py"
)
validation_dir = materialize_runtime_directory(
    "validator_raw", "44b6_12dfb391.geff/zarr.json"
)

wheel_dirs = sorted({p.parent for p in support.rglob("*.whl")})
if not wheel_dirs:
    raise FileNotFoundError(f"No offline wheels found below {support}")
pip_cmd = [sys.executable, "-m", "pip", "install", "--no-index"]
for wheel_dir in wheel_dirs:
    pip_cmd.extend(["--find-links", str(wheel_dir)])
pip_cmd.extend([
    "numpy==2.0.2",
    "tracksdata",
    "zarr>=3.0.10,<4",
    "polars>=1.36",
    "blosc2",
    "dask[array]",
    "pyyaml",
])
print("Installing verified offline dependencies from", support)
subprocess.run(pip_cmd, check=True)
for module in ("tracksdata", "zarr", "polars", "dask", "yaml"):
    importlib.import_module(module)

manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
expected_model_hash = manifest["pretrained_model"]["model_sha256"]
actual_model_hash = hashlib.sha256((ctc_dir / "model.pt").read_bytes()).hexdigest()
if actual_model_hash != expected_model_hash:
    raise RuntimeError(f"Pretrained model hash mismatch: {actual_model_hash}")
print(json.dumps({
    "runtime": str(runtime),
    "competition": str(competition),
    "support": str(support),
    "model_sha256": actual_model_hash,
}, indent=2))
'''


TRAIN = r'''output_dir = Path("/kaggle/working/trackastra_graph_v2")
command = [
    sys.executable,
    str(runtime / "trainer.py"),
    "--competition-dir", str(competition),
    "--trackastra-dir", str(trackastra_dir),
    "--pretrained-dir", str(ctc_dir),
    "--validation-predictions", str(validation_dir),
    "--output-dir", str(output_dir),
    "--steps", "6000",
    "--train-per-prefix", "24",
    "--max-tokens", "512",
    "--gradient-accumulation", "4",
    "--validation-samples", "24",
    "--max-wall-seconds", "13200",
    "--validation-reserve-seconds", "1800",
]
print("Launching Biohub-native Trackastra fine-tuning:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as exc:
    write_terminal("failed", exc)
    raise

training_terminal = output_dir / "training_terminal.json"
if not training_terminal.is_file():
    raise RuntimeError("Trainer exited without terminal evidence")
result = json.loads(training_terminal.read_text(encoding="utf-8"))
print(json.dumps({
    "step": result["step"],
    "elapsed_seconds": result["elapsed_seconds"],
    "delta_proxy_vs_public_0927_baseline": result["delta_proxy_vs_public_0927_baseline"],
    "best_summary": result["complete_movie_best"]["summary"],
}, indent=2))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Experiment complete; evidence is in", Path("/kaggle/working/trackastra_graph_v2"))
'''


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
                "# Biohub-native Trackastra association fine-tuning\n\n"
                "This is a new candidate trained on Biohub graph supervision. Public 0.927 outputs "
                "are used only as frozen detector inputs for clean validation.\n"
            ),
            code_cell(SETUP),
            code_cell(TRAIN),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": "indarkarhana/biohub-trackastra-graph-finetune-v1",
        "title": "Biohub Trackastra Graph Finetune v2",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "transformer"],
        "dataset_sources": [
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
