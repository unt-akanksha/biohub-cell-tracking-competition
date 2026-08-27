from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-trackastra-dual-fold-synthetic-v1"
NOTEBOOK = TARGET / "biohub-trackastra-dual-fold-synthetic-v1.ipynb"


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

RUN_ID = "trackastra-dual-fold-synthetic-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    training_terminal = Path("/kaggle/working/trackastra_dual_fold_v1/training_terminal.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 14400,
        "hard_stop_seconds": 14100,
        "training_terminal_exists": training_terminal.is_file(),
        "submission_created": False,
    }
    if error:
        payload["error"] = str(error)
    if training_terminal.is_file():
        payload["training_terminal_sha256"] = hashlib.sha256(
            training_terminal.read_bytes()
        ).hexdigest()
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
print("Dual-fold watchdog armed for 14,100 seconds.")
'''


SETUP = r'''import importlib
import shutil
import subprocess
import sys
import zipfile

import torch


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"This experiment requires exactly two GPUs, saw {torch.cuda.device_count()}"
    )
print({"gpu_count": 2, "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)]})

input_root = Path("/kaggle/input")
runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-trackastra-dual-runtime-v1"),
    Path("/kaggle/input/biohub-trackastra-dual-runtime-v1"),
])
competition = first_existing([
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
])
support = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
])
synthetic = next(
    (
        path.parent
        for path in input_root.rglob("manifest.json")
        if (path.parent / "sequences").is_dir()
    ),
    None,
)
if runtime is None:
    runtime = next(
        (path for path in input_root.iterdir() if (path / "dual_trainer.py").is_file()),
        None,
    )
if support is None:
    support = next((path for path in input_root.iterdir() if (path / "wheels").is_dir()), None)
if runtime is None or competition is None or support is None or synthetic is None:
    raise FileNotFoundError({
        "runtime": runtime,
        "competition": competition,
        "support": support,
        "synthetic": synthetic,
    })

runtime_bundle = runtime / "runtime_bundle.zip"
if runtime_bundle.is_file():
    materialized_runtime = Path("/kaggle/working/runtime_bundle")
    if materialized_runtime.exists():
        shutil.rmtree(materialized_runtime)
    materialized_runtime.mkdir(parents=True)
    with zipfile.ZipFile(runtime_bundle) as handle:
        handle.extractall(materialized_runtime)
    runtime = materialized_runtime
if not (runtime / "SOURCE_MANIFEST.json").is_file():
    raise FileNotFoundError(f"runtime manifest missing: {runtime}")


def materialize_runtime_directory(name, marker):
    direct = runtime / name
    if (direct / marker).exists():
        return direct
    archive = runtime / f"{name}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"missing runtime directory/archive: {name}")
    destination = Path("/kaggle/working/runtime_materialized") / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    for candidate in (destination, destination / name):
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"archive {archive} did not contain {marker}")


ctc_dir = materialize_runtime_directory("ctc", "model.pt")
trackastra_dir = materialize_runtime_directory("trackastra_source", "trackastra/model/model.py")
wheel_dirs = sorted({path.parent for path in support.rglob("*.whl")})
if not wheel_dirs:
    raise FileNotFoundError(f"no offline wheels found below {support}")
numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"unexpected Kaggle NumPy before install: {numpy_before}")
pip_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
for wheel_dir in wheel_dirs:
    pip_cmd.extend(["--find-links", str(wheel_dir)])
pip_cmd.extend([
    "bidict==0.23.1",
    "donfig==0.8.1.post1",
    "geff==1.2.0.1.1",
    "geff-spec==1.1.1",
    "ilpy==0.6.0",
    "numcodecs==0.15.1",
    "polars==1.42.0",
    "polars-runtime-32==1.42.0",
    "pyscipopt==6.2.1",
    "rustworkx==0.18.0",
    "tracksdata==0.1.0rc6.dev3+g980c2d30a",
    "zarr==3.2.1",
])
subprocess.run(pip_cmd, check=True)
for module in ("numpy", "scipy", "tracksdata", "zarr", "polars", "dask", "yaml"):
    importlib.import_module(module)
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("offline install changed NumPy")

manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
expected_model_hash = manifest["pretrained_model"]["model_sha256"]
actual_model_hash = hashlib.sha256((ctc_dir / "model.pt").read_bytes()).hexdigest()
if actual_model_hash != expected_model_hash:
    raise RuntimeError(f"pretrained model hash mismatch: {actual_model_hash}")
if manifest["integrity"]["submission_created"] is not False:
    raise RuntimeError("runtime integrity policy changed")
print(json.dumps({
    "runtime": str(runtime),
    "competition": str(competition),
    "support": str(support),
    "synthetic": str(synthetic),
    "model_sha256": actual_model_hash,
}, indent=2))
'''


TRAIN = r'''output_dir = Path("/kaggle/working/trackastra_dual_fold_v1")
command = [
    sys.executable,
    str(runtime / "dual_trainer.py"),
    "--orchestrate",
    "--competition-dir", str(competition),
    "--trackastra-dir", str(trackastra_dir),
    "--pretrained-dir", str(ctc_dir),
    "--synthetic-root", str(synthetic),
    "--output-dir", str(output_dir),
    "--steps", "75000",
    "--max-wall-seconds", "12000",
    "--orchestrator-hard-stop-seconds", "13200",
    "--finalization-reserve-seconds", "900",
    "--synthetic-train-graphs", "2046",
    "--synthetic-validation-graphs", "128",
    "--real-train-graphs", "96",
    "--real-validation-graphs", "12",
    "--max-tokens", "512",
    "--real-replay-probability", "0.05",
]
print("Launching two isolated reciprocal-fold workers:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as exc:
    write_terminal("failed", exc)
    raise

training_terminal = output_dir / "training_terminal.json"
if not training_terminal.is_file():
    raise RuntimeError("trainer exited without terminal evidence")
result = json.loads(training_terminal.read_text(encoding="utf-8"))
if result["gpu_count"] != 2 or sorted(result["whole_fold_coverage"]) != [
    "target_44b6", "target_6bba"
]:
    raise RuntimeError("dual-fold coverage evidence failed")
if result["submission_created"] is not False:
    raise RuntimeError("training run unexpectedly created a submission")
print(json.dumps(result, indent=2))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Training-only experiment complete; no submission was created.")
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
                "# Two-GPU reciprocal-fold Trackastra adaptation\n\n"
                "Two independent 27.5M-parameter models train on corrected CC0 synthetic graphs "
                "with small opposite-embryo real replay. Fixed unopened internal movies select "
                "checkpoints. This notebook cannot create or submit a competition archive.\n"
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
        "id": "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
        "title": "Biohub Trackastra Dual Fold Synthetic v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "transformer", "synthetic-data"],
        "dataset_sources": [
            "indarkarhana/biohub-trackastra-dual-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": ["josefreitasalvesneto/biohub-synthetic-dataset"],
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
