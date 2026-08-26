from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-spatialdino-appearance-validation-v1"
NOTEBOOK = TARGET / "biohub-spatialdino-appearance-validation-v1.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).lstrip().splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip().splitlines(keepends=True),
    }


WATCHDOG = r'''
import atexit
import hashlib
import json
import os
import threading
import time
from pathlib import Path

RUN_ID = "spatialdino-appearance-validation-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    training = Path("/kaggle/working/spatialdino_appearance_v1/training_terminal.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 7200,
        "hard_stop_seconds": 6900,
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
TIMER = threading.Timer(6900, budget_expired)
TIMER.daemon = True
TIMER.start()
print("SpatialDINO appearance watchdog armed for 6,900 seconds.")
'''


SETUP = r'''
import importlib
import shutil
import subprocess
import sys
import zipfile

CHECKPOINT_SHA256 = "47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8"
PROCESSED_VALIDATION_SHA256 = "6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b"
HOCT_TRAINING_TERMINAL_SHA256 = "8ddd87ca0dd748b668e8c4f61409930d071ef84f9dd82d96fd750e8b98943f69"


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


input_root = Path("/kaggle/input")
runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-spatialdino-runtime-v1"),
    Path("/kaggle/input/biohub-spatialdino-runtime-v1"),
])
hoct_runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-hoct-multibackbone-runtime-v1"),
    Path("/kaggle/input/biohub-hoct-multibackbone-runtime-v1"),
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
hoct_output = first_existing([
    Path("/kaggle/input/biohub-hoct-multibackbone-probe-v1"),
    Path("/kaggle/input/kernels/indarkarhana/biohub-hoct-multibackbone-probe-v1"),
])
if None in (runtime, hoct_runtime, graph_runtime, competition, support, hoct_output):
    raise FileNotFoundError({
        "runtime": runtime,
        "hoct_runtime": hoct_runtime,
        "graph_runtime": graph_runtime,
        "competition": competition,
        "support": support,
        "hoct_output": hoct_output,
    })


def materialize_root(source, name):
    bundle = source / "runtime_bundle.zip"
    if not bundle.is_file():
        return source
    destination = Path("/kaggle/working") / f"{name}_runtime_bundle"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as handle:
        handle.extractall(destination)
    return destination


graph_runtime = materialize_root(graph_runtime, "graph")


def materialize_runtime_directory(source, name, marker):
    direct = source / name
    if (direct / marker).exists():
        return direct
    archive = source / f"{name}.zip"
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
    "rustworkx==0.18.0", "tracksdata==0.1.0rc6.dev3+g980c2d30a", "zarr==3.2.1",
])
subprocess.run(pip_cmd, check=True)
for module in ("numpy", "scipy", "torch", "tracksdata", "zarr", "polars"):
    importlib.import_module(module)
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("Offline dependency install changed NumPy")
if not importlib.import_module("torch").cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

runtime_manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in (
    "appearance.py", "correction.py", "encoder.py",
    "validate_appearance_correction.py", "spatialdino_vits8_backbone.pth",
):
    actual = hashlib.sha256((runtime / name).read_bytes()).hexdigest()
    if actual != runtime_manifest["files"][name]["sha256"]:
        raise RuntimeError(f"SpatialDINO runtime hash mismatch: {name}")
if runtime_manifest["pretrained_model"]["sha256"] != CHECKPOINT_SHA256:
    raise RuntimeError("SpatialDINO checkpoint manifest mismatch")

hoct_manifest = json.loads((hoct_runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in ("train_biohub_hoct_probe.py",):
    actual = hashlib.sha256((hoct_runtime / name).read_bytes()).hexdigest()
    if actual != hoct_manifest["files"][name]["sha256"]:
        raise RuntimeError(f"HOCT scoring runtime hash mismatch: {name}")
graph_manifest = json.loads((graph_runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in ("trainer.py", "hybrid_linker.py", "rerank_submission.py"):
    actual = hashlib.sha256((graph_runtime / name).read_bytes()).hexdigest()
    if actual != graph_manifest["files"][name]["sha256"]:
        raise RuntimeError(f"Graph runtime hash mismatch: {name}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


valid_topologies = []
for launcher_path in hoct_output.rglob("launcher_terminal.json"):
    try:
        launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if not (
        launcher.get("run_id") == "hoct-multibackbone-probe-v1"
        and launcher.get("status") == "completed"
        and launcher.get("submission_created") is False
        and launcher.get("training_terminal_sha256") == HOCT_TRAINING_TERMINAL_SHA256
    ):
        continue
    for csv_path in launcher_path.parent.rglob("processed_validation.csv"):
        if sha256_file(csv_path) == PROCESSED_VALIDATION_SHA256:
            valid_topologies.append(csv_path)
if len(valid_topologies) != 1:
    raise FileNotFoundError(f"Expected one hash-bound processed topology, found {valid_topologies}")
processed_validation_csv = valid_topologies[0]
print(json.dumps({
    "runtime": str(runtime),
    "competition": str(competition),
    "validation_dir": str(validation_dir),
    "processed_validation_csv": str(processed_validation_csv),
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "cuda_devices": importlib.import_module("torch").cuda.device_count(),
}, indent=2))
'''


RUN = r'''
output_dir = Path("/kaggle/working/spatialdino_appearance_v1")
command = [
    sys.executable,
    str(runtime / "validate_appearance_correction.py"),
    "--checkpoint", str(runtime / "spatialdino_vits8_backbone.pth"),
    "--competition-dir", str(competition),
    "--raw-validation-root", str(validation_dir),
    "--processed-validation-csv", str(processed_validation_csv),
    "--output-dir", str(output_dir),
    "--batch-size", "4",
    "--max-wall-seconds", "6600",
]
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join([
    str(runtime), str(hoct_runtime), str(graph_runtime)
])
print("Launching SpatialDINO appearance validation:", " ".join(command))
try:
    subprocess.run(command, check=True, env=environment)
except Exception as exc:
    write_terminal("failed", exc)
    raise

training = json.loads((output_dir / "training_terminal.json").read_text(encoding="utf-8"))
if Path("/kaggle/working/submission.csv").exists():
    raise RuntimeError("Validation unexpectedly created a submission")
print(json.dumps({
    "selection_delta_vs_base": training["selection_delta_vs_base"],
    "selection_passed": training["selection_passed"],
    "acceptance_skipped": training["acceptance_skipped"],
    "acceptance_delta_vs_base": training["acceptance_delta_vs_base"],
    "association_acceptance_passed": training["association_acceptance_passed"],
}, indent=2))
'''


FINISH = r'''
FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("SpatialDINO clean validation complete; no submission was created.")
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
                """
                # SpatialDINO degree-preserving appearance correction

                This private clean-validation experiment freezes the official
                SpatialDINO ViT-S/8 backbone and the audited detector topology.
                It can only exchange two ambiguous ordinary parent links while
                preserving every node, edge count, and in/out degree. Selection
                and acceptance use disjoint complete movies, no leaderboard
                feedback is read, and no submission is created.
                """
            ),
            code_cell(SETUP),
            code_cell(RUN),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": "indarkarhana/biohub-spatialdino-appearance-validation-v1",
        "title": "Biohub SpatialDINO Appearance Validation v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "spatialdino", "appearance"],
        "dataset_sources": [
            "indarkarhana/biohub-spatialdino-runtime-v1",
            "indarkarhana/biohub-hoct-multibackbone-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-hoct-multibackbone-probe-v1"
        ],
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
