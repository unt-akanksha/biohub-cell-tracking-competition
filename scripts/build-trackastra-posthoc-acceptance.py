from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-trackastra-raw-confidence-acceptance-v1"
NOTEBOOK = TARGET / "biohub-trackastra-raw-confidence-acceptance-v1.ipynb"


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

RUN_ID = "trackastra-raw-confidence-acceptance-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    acceptance = Path("/kaggle/working/trackastra_acceptance/acceptance_terminal.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 5400,
        "hard_stop_seconds": 5100,
        "acceptance_terminal_exists": acceptance.is_file(),
        "submission_created": Path("/kaggle/working/submission.csv").is_file(),
    }
    if error:
        payload["error"] = str(error)
    if acceptance.is_file():
        payload["acceptance_terminal_sha256"] = hashlib.sha256(
            acceptance.read_bytes()
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
TIMER = threading.Timer(5100, budget_expired)
TIMER.daemon = True
TIMER.start()
print("Posthoc acceptance watchdog armed for 5,100 seconds.")
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
deepcenter = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),
    Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),
])
if runtime is None:
    runtime = next(
        (
            path
            for path in input_root.iterdir()
            if (path / "SOURCE_MANIFEST.json").is_file()
            or (path / "runtime_bundle.zip").is_file()
        ),
        None,
    )
if support is None:
    support = next((path for path in input_root.iterdir() if (path / "wheels").is_dir()), None)
if runtime is None or competition is None or support is None or deepcenter is None:
    raise FileNotFoundError({
        "runtime": runtime,
        "competition": competition,
        "support": support,
        "deepcenter": deepcenter,
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


trackastra_dir = materialize_runtime_directory(
    "trackastra_source", "trackastra/model/model.py"
)
validation_dir = materialize_runtime_directory(
    "validator_raw", "44b6_12dfb391.geff/zarr.json"
)

wheel_dirs = sorted({path.parent for path in support.rglob("*.whl")})
if not wheel_dirs:
    raise FileNotFoundError(f"No offline wheels found below {support}")
numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"Unexpected Kaggle NumPy before offline install: {numpy_before}")
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
for module in ("numpy", "scipy", "tracksdata", "zarr", "polars", "dask", "yaml"):
    importlib.import_module(module)
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("Offline dependency install changed NumPy")

manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in (
    "trainer.py",
    "hybrid_linker.py",
    "rerank_submission.py",
    "validate_model_posthoc.py",
    "materialize_public_validation.py",
    "public_config_source.py",
    "public_postprocess_source.py",
):
    actual = hashlib.sha256((runtime / name).read_bytes()).hexdigest()
    if actual != manifest["files"][name]["sha256"]:
        raise RuntimeError(f"Runtime source hash mismatch for {name}")

training_candidates = []
for path in input_root.rglob("training_terminal.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if payload.get("status") != "completed" or "model_sha256" not in payload:
        continue
    model_path = path.parent / "model.pt"
    if not model_path.is_file():
        continue
    if hashlib.sha256(model_path.read_bytes()).hexdigest() == payload["model_sha256"]:
        training_candidates.append((path, payload))
if len(training_candidates) != 1:
    raise FileNotFoundError(
        f"Expected one hash-valid V6 training output, found {[str(path) for path, _ in training_candidates]}"
    )
training_terminal, training_payload = training_candidates[0]
model_dir = training_terminal.parent
if not __import__("torch").cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print(json.dumps({
    "runtime": str(runtime),
    "validation_dir": str(validation_dir),
    "model_dir": str(model_dir),
    "model_sha256": training_payload["model_sha256"],
    "source_training_acceptance_passed": training_payload.get("association_acceptance_passed"),
    "deepcenter": str(deepcenter),
}, indent=2))
'''


EVALUATE = r'''output_dir = Path("/kaggle/working/trackastra_acceptance")
processed_dir = Path("/kaggle/working/processed_validation")
materialize_command = [
    sys.executable,
    str(runtime / "materialize_public_validation.py"),
    "--raw-validation-root", str(validation_dir),
    "--competition-dir", str(competition),
    "--public-config-source", str(runtime / "public_config_source.py"),
    "--public-postprocess-source", str(runtime / "public_postprocess_source.py"),
    "--output-dir", str(processed_dir),
]
print("Materializing frozen processed comparator:", " ".join(materialize_command))
try:
    subprocess.run(materialize_command, check=True)
except Exception as exc:
    write_terminal("failed", exc)
    raise

command = [
    sys.executable,
    str(runtime / "validate_model_posthoc.py"),
    "--model-dir", str(model_dir),
    "--trackastra-dir", str(trackastra_dir),
    "--validation-predictions", str(validation_dir),
    "--processed-validation-csv", str(processed_dir / "processed_validation.csv"),
    "--competition-dir", str(competition),
    "--output-dir", str(output_dir),
    "--max-tokens", "512",
    "--candidate-radius", "80",
]
print("Launching frozen posthoc acceptance:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as exc:
    write_terminal("failed", exc)
    raise

terminal = json.loads((output_dir / "acceptance_terminal.json").read_text(encoding="utf-8"))
if Path("/kaggle/working/submission.csv").exists():
    raise RuntimeError("Acceptance unexpectedly created a submission")
print(json.dumps({
    "selected_method": terminal["complete_movie_selected"]["method"],
    "selection_delta_vs_base_raw": terminal["selection_delta_vs_base_raw"],
    "acceptance_delta_vs_base_raw": terminal["acceptance_delta_vs_base_raw"],
    "association_acceptance_passed": terminal["association_acceptance_passed"],
}, indent=2))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Frozen posthoc acceptance complete; no submission was created.")
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
                "# Trackastra probability-aware frozen acceptance\n\n"
                "This evaluation selects association rules on one complete movie per embryo, "
                "then reads the other two movies exactly once after configuration freeze. It "
                "uses stored pre-postprocessing edge confidence and creates no submission.\n"
            ),
            code_cell(SETUP),
            code_cell(EVALUATE),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v1",
        "title": "Biohub Trackastra Raw Confidence Acceptance v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "validation"],
        "dataset_sources": [
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        ],
        "kernel_sources": ["indarkarhana/biohub-trackastra-graph-finetune-v6"],
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
