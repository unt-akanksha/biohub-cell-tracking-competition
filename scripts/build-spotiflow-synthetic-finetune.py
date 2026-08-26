from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-spotiflow-synthetic-finetune-v1"
NOTEBOOK = TARGET / "biohub-spotiflow-synthetic-finetune-v1.ipynb"


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

RUN_ID = "spotiflow-synthetic-finetune-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    result = Path("/kaggle/working/spotiflow_synthetic_finetune/synthetic_finetune_result.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 7200,
        "hard_stop_seconds": 6900,
        "result_exists": result.is_file(),
    }
    if error:
        payload["error"] = str(error)
    if result.is_file():
        payload["result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
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
TIMER = threading.Timer(6900, budget_expired)
TIMER.daemon = True
TIMER.start()
print("Synthetic detector fine-tune watchdog armed for 6,900 seconds.")
'''


SETUP = r'''import importlib
import subprocess
import sys


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


input_root = Path("/kaggle/input")
runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-spotiflow-detector-runtime-v1"),
    Path("/kaggle/input/biohub-spotiflow-detector-runtime-v1"),
])
synthetic = next(
    (
        path.parent
        for path in input_root.rglob("manifest.json")
        if (path.parent / "static").is_dir()
    ),
    None,
)
if runtime is None:
    runtime = next((p for p in input_root.iterdir() if (p / "models/synth_3d/best.pt").is_file()), None)
if runtime is None or synthetic is None:
    raise FileNotFoundError({"runtime": runtime, "synthetic": synthetic})

numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"Unexpected Kaggle NumPy before offline install: {numpy_before}")
runtime_wheels = sorted((runtime / "wheels").glob("*.whl"))
if not runtime_wheels:
    raise FileNotFoundError(f"No Spotiflow wheels found under {runtime}")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, runtime_wheels)],
    check=True,
)
for module in ("numpy", "torch", "lightning", "spotiflow"):
    imported = importlib.import_module(module)
    print(module, getattr(imported, "__version__", "unknown"))
numpy_after = importlib.import_module("numpy").__version__
if numpy_after != numpy_before:
    raise RuntimeError(f"Offline install changed NumPy: {numpy_before} -> {numpy_after}")
if not importlib.import_module("torch").cuda.is_available():
    raise RuntimeError("CUDA is not available")

manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in ("train_synthetic_detector.py", "synthetic_data.py"):
    path = runtime / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest["scripts"][name]:
        raise RuntimeError(f"runtime script hash mismatch for {name}: {actual}")
print(json.dumps({"runtime": str(runtime), "synthetic": str(synthetic)}, indent=2))
'''


RUN = r'''BASE_MODEL = "synth_3d"
output_dir = Path("/kaggle/working/spotiflow_synthetic_finetune")
command = [
    sys.executable,
    str(runtime / "train_synthetic_detector.py"),
    "--synthetic-root", str(synthetic),
    "--pretrained-model", str(runtime / "models" / BASE_MODEL),
    "--output-dir", str(output_dir),
    "--validation-count", "128",
    "--train-limit", "1200",
    "--epochs", "4",
    "--samples-per-epoch", "768",
    "--learning-rate", "3e-5",
    "--seed", "20260826",
    "--max-wall-seconds", "6300",
]
print("Launching synthetic detector fine-tune:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as exc:
    write_terminal("failed", exc)
    raise

result = output_dir / "synthetic_finetune_result.json"
best = output_dir / "best.pt"
if not result.is_file() or not best.is_file():
    raise RuntimeError("Trainer exited without result and best checkpoint")
summary = json.loads(result.read_text(encoding="utf-8"))
if not summary["weights_changed"] or summary["parameter_count"] != 35489892:
    raise RuntimeError(f"Invalid fine-tune result: {summary}")
print(json.dumps(summary, indent=2, sort_keys=True))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Synthetic detector fine-tune complete; no competition submission was created.")
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
                "# Biohub Spotiflow synthetic fine-tune\n\n"
                "Independent 35.5M-parameter detector training on the audited CC0 synthetic source. "
                "Static image and centroid coordinates are pooled together; no competition output is created.\n"
            ),
            code_cell(SETUP),
            code_cell(RUN),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": "indarkarhana/biohub-spotiflow-synthetic-finetune-v1",
        "title": "Biohub Spotiflow Synthetic Finetune v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-detection", "spotiflow", "synthetic-data"],
        "dataset_sources": ["indarkarhana/biohub-spotiflow-detector-runtime-v1"],
        "kernel_sources": ["josefreitasalvesneto/biohub-synthetic-dataset"],
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
