from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-hoct-probe-finetune-v1"
NOTEBOOK = TARGET / "biohub-hoct-probe-finetune-v1.ipynb"


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

RUN_ID = "hoct-probe-finetune-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    training = Path("/kaggle/working/hoct_probe_v1/training_terminal.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 14400,
        "hard_stop_seconds": 14100,
        "training_terminal_exists": training.is_file(),
        "submission_created": Path("/kaggle/working/submission.csv").is_file(),
    }
    if error:
        payload["error"] = str(error)
    if training.is_file():
        payload["training_terminal_sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
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
print("HOCT probe watchdog armed for 14,100 seconds.")
'''


SETUP = r'''import importlib
import shutil
import subprocess
import sys
import zipfile


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


input_root = Path("/kaggle/input")
hoct_runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-hoct-runtime-v1"),
    Path("/kaggle/input/biohub-hoct-runtime-v1"),
])
graph_runtime = first_existing([
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
deepcenter = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),
    Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),
])
if None in (hoct_runtime, graph_runtime, competition, support, deepcenter):
    raise FileNotFoundError({
        "hoct_runtime": hoct_runtime,
        "graph_runtime": graph_runtime,
        "competition": competition,
        "support": support,
        "deepcenter": deepcenter,
    })


def materialize_root(runtime, name):
    bundle = runtime / "runtime_bundle.zip"
    if not bundle.is_file():
        return runtime
    destination = Path("/kaggle/working") / f"{name}_runtime_bundle"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as handle:
        handle.extractall(destination)
    return destination


hoct_runtime = materialize_root(hoct_runtime, "hoct")
graph_runtime = materialize_root(graph_runtime, "graph")


def materialize_runtime_directory(runtime, name, marker):
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


validation_dir = materialize_runtime_directory(
    graph_runtime, "validator_raw", "44b6_12dfb391.geff/zarr.json"
)
wheel_dirs = sorted({path.parent for path in support.rglob("*.whl")})
if not wheel_dirs:
    raise FileNotFoundError(f"No offline wheels found below {support}")
numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"Unexpected Kaggle NumPy: {numpy_before}")
pip_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
for wheel_dir in wheel_dirs:
    pip_cmd.extend(["--find-links", str(wheel_dir)])
pip_cmd.extend([
    "bidict==0.23.1", "donfig==0.8.1.post1", "geff==1.2.0.1.1",
    "geff-spec==1.1.1", "ilpy==0.6.0", "numcodecs==0.15.1",
    "polars==1.42.0", "polars-runtime-32==1.42.0", "pyscipopt==6.2.1",
    "rustworkx==0.18.0", "tracksdata==0.1.0rc6.dev3+g980c2d30a",
    "zarr==3.2.1",
])
subprocess.run(pip_cmd, check=True)
for module in ("numpy", "scipy", "torch", "tracksdata", "zarr", "polars", "dask", "yaml"):
    importlib.import_module(module)
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("Offline dependency install changed NumPy")
if not importlib.import_module("torch").cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

hoct_manifest = json.loads((hoct_runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in ("biohub_adapter.py", "train_biohub_hoct_probe.py", "general_v1.pt"):
    actual = hashlib.sha256((hoct_runtime / name).read_bytes()).hexdigest()
    if actual != hoct_manifest["files"][name]["sha256"]:
        raise RuntimeError(f"HOCT runtime hash mismatch: {name}")
graph_manifest = json.loads((graph_runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in (
    "trainer.py", "hybrid_linker.py", "rerank_submission.py",
    "materialize_public_validation.py", "public_config_source.py",
    "public_postprocess_source.py",
):
    actual = hashlib.sha256((graph_runtime / name).read_bytes()).hexdigest()
    if actual != graph_manifest["files"][name]["sha256"]:
        raise RuntimeError(f"Graph runtime hash mismatch: {name}")
print(json.dumps({
    "hoct_runtime": str(hoct_runtime),
    "graph_runtime": str(graph_runtime),
    "competition": str(competition),
    "validation_dir": str(validation_dir),
    "deepcenter": str(deepcenter),
    "source_model_sha256": hoct_manifest["pretrained_model"]["sha256"],
}, indent=2))
'''


TRAIN = r'''processed_dir = Path("/kaggle/working/processed_validation")
materialize_command = [
    sys.executable,
    str(graph_runtime / "materialize_public_validation.py"),
    "--raw-validation-root", str(validation_dir),
    "--competition-dir", str(competition),
    "--public-config-source", str(graph_runtime / "public_config_source.py"),
    "--public-postprocess-source", str(graph_runtime / "public_postprocess_source.py"),
    "--output-dir", str(processed_dir),
]
print("Materializing final-topology clean validation:", " ".join(materialize_command))
try:
    subprocess.run(materialize_command, check=True)
except Exception as exc:
    write_terminal("failed", exc)
    raise

output_dir = Path("/kaggle/working/hoct_probe_v1")
command = [
    sys.executable,
    str(hoct_runtime / "train_biohub_hoct_probe.py"),
    "--competition-dir", str(competition),
    "--pretrained-model", str(hoct_runtime / "general_v1.pt"),
    "--validation-predictions", str(validation_dir),
    "--processed-validation-csv", str(processed_dir / "processed_validation.csv"),
    "--output-dir", str(output_dir),
    "--train-per-prefix", "128",
    "--core-size", "128",
    "--max-wall-seconds", "13200",
]
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join([str(hoct_runtime), str(graph_runtime)])
print("Launching Biohub HOCT probe:", " ".join(command))
try:
    subprocess.run(command, check=True, env=environment)
except Exception as exc:
    write_terminal("failed", exc)
    raise

training = json.loads((output_dir / "training_terminal.json").read_text(encoding="utf-8"))
if Path("/kaggle/working/submission.csv").exists():
    raise RuntimeError("Training unexpectedly created a submission")
print(json.dumps({
    "feature_extraction": training["feature_extraction"],
    "selected": training["selected"],
    "selection_delta_vs_base": training["selection_delta_vs_base"],
    "acceptance_delta_vs_base": training["acceptance_delta_vs_base"],
    "association_acceptance_passed": training["association_acceptance_passed"],
}, indent=2))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("HOCT probe experiment complete; no submission was created.")
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
                "# Biohub-supervised Higher-Order Cell Tracking Transformer\n\n"
                "This independent association lane uses the official MIT-licensed HOCT "
                "general model, fits only a 289-parameter linear probe on Biohub ground-truth "
                "graphs, and evaluates final detector topology with a disjoint clean "
                "selection/acceptance split. It creates no competition submission.\n"
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
        "id": "indarkarhana/biohub-hoct-probe-finetune-v1",
        "title": "Biohub HOCT Probe Finetune v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "transformer", "validation"],
        "dataset_sources": [
            "indarkarhana/biohub-hoct-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        ],
        "kernel_sources": [],
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
