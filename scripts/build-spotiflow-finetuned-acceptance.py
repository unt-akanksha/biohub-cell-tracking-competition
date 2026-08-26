from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-spotiflow-finetuned-acceptance-v1"
NOTEBOOK = TARGET / "biohub-spotiflow-finetuned-acceptance-v1.ipynb"


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

RUN_ID = "spotiflow-finetuned-acceptance-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    result = Path("/kaggle/working/spotiflow_finetuned_acceptance/finetuned_detector_acceptance.json")
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
print("Fine-tuned detector acceptance watchdog armed for 6,900 seconds.")
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
training_result = next(input_root.rglob("synthetic_finetune_result.json"), None)
candidate = training_result.parent if training_result is not None else None
if runtime is None:
    runtime = next((p for p in input_root.iterdir() if (p / "evaluate_finetuned_detector.py").is_file()), None)
if graph_runtime is None:
    graph_runtime = next((p for p in input_root.iterdir() if (p / "validator_raw/44b6_12dfb391.geff/zarr.json").is_file()), None)
if support is None:
    support = next((p for p in input_root.iterdir() if (p / "wheels").is_dir() and p != runtime), None)
if None in (runtime, graph_runtime, competition, support, training_result, candidate):
    raise FileNotFoundError({
        "runtime": runtime,
        "graph_runtime": graph_runtime,
        "competition": competition,
        "support": support,
        "training_result": training_result,
    })

support_wheels = sorted({p.parent for p in support.rglob("*.whl")})
if not support_wheels:
    raise FileNotFoundError(f"No support wheels found under {support}")
numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"Unexpected Kaggle NumPy before offline install: {numpy_before}")

base_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
for wheel_dir in support_wheels:
    base_cmd.extend(["--find-links", str(wheel_dir)])
base_cmd.extend([
    "bidict==0.23.1", "donfig==0.8.1.post1", "geff==1.2.0.1.1",
    "geff-spec==1.1.1", "ilpy==0.6.0", "numcodecs==0.15.1",
    "polars==1.42.0", "polars-runtime-32==1.42.0", "pyscipopt==6.2.1",
    "rustworkx==0.18.0", "tracksdata==0.1.0rc6.dev3+g980c2d30a", "zarr==3.2.1",
])
subprocess.run(base_cmd, check=True)
runtime_wheels = sorted((runtime / "wheels").glob("*.whl"))
if not runtime_wheels:
    raise FileNotFoundError(f"No Spotiflow wheels found under {runtime}")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, runtime_wheels)],
    check=True,
)

support_repo = next((p for p in support.rglob("biohub_tracking") if p.is_dir()), None)
if support_repo is None:
    raise FileNotFoundError("biohub_tracking package source was not found")
sys.path.insert(0, str(support_repo.parent))
for module in ("numpy", "scipy", "torch", "tracksdata", "zarr", "spotiflow"):
    imported = importlib.import_module(module)
    print(module, getattr(imported, "__version__", "unknown"))
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("Offline install changed NumPy")
from scipy.optimize import linear_sum_assignment
linear_sum_assignment([[0.0, 1.0], [1.0, 0.0]])

probe = importlib.import_module("tracksdata").graph.IndexedRXGraph.from_geff(
    graph_runtime / "validator_raw" / "44b6_12dfb391.geff"
)
probe = probe[0] if isinstance(probe, tuple) else probe
if probe.node_attrs().height <= 0 or probe.edge_attrs().height <= 0:
    raise RuntimeError("Offline graph-IO probe returned an empty validation graph")

runtime_manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
evaluator = runtime / "evaluate_finetuned_detector.py"
if hashlib.sha256(evaluator.read_bytes()).hexdigest() != runtime_manifest["scripts"][evaluator.name]:
    raise RuntimeError("fine-tuned evaluator hash mismatch")
learned = json.loads(training_result.read_text(encoding="utf-8"))
best = candidate / "best.pt"
if hashlib.sha256(best.read_bytes()).hexdigest() != learned["best_weight_sha256"]:
    raise RuntimeError("learned checkpoint hash mismatch")
print(json.dumps({
    "runtime": str(runtime),
    "graph_runtime": str(graph_runtime),
    "competition": str(competition),
    "support": str(support),
    "candidate": str(candidate),
}, indent=2))
'''


RUN = r'''output_dir = Path("/kaggle/working/spotiflow_finetuned_acceptance")
command = [
    sys.executable,
    str(runtime / "evaluate_finetuned_detector.py"),
    "--competition-dir", str(competition),
    "--model-dir", str(candidate),
    "--training-result", str(training_result),
    "--baseline-predictions", str(graph_runtime / "validator_raw"),
    "--output-dir", str(output_dir),
    "--calibration-frames", "12",
    "--pretrained-acceptance-recall", "0.7776834959694527",
    "--max-wall-seconds", "6300",
]
print("Launching learned detector acceptance:", " ".join(command))
try:
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = os.pathsep.join(
        [str(support_repo.parent), run_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(command, check=True, env=run_env)
except Exception as exc:
    write_terminal("failed", exc)
    raise

result = output_dir / "finetuned_detector_acceptance.json"
if not result.is_file():
    raise RuntimeError("Evaluator exited without fine-tuned acceptance result")
summary = json.loads(result.read_text(encoding="utf-8"))
print(json.dumps(summary["acceptance"], indent=2, sort_keys=True))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Fine-tuned detector acceptance complete; no submission was created.")
'''


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {
                "accelerator": "gpu", "dataSources": [], "isInternetEnabled": False,
                "language": "python", "sourceType": "notebook", "isGpuEnabled": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            code_cell(WATCHDOG),
            markdown_cell(
                "# Learned Spotiflow detector acceptance\n\n"
                "Fixed auto normalization and metadata-only threshold calibration on four complete "
                "heldout movies. Ground truth is read only after each threshold is fixed. No submission.\n"
            ),
            code_cell(SETUP), code_cell(RUN), code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": "indarkarhana/biohub-spotiflow-finetuned-acceptance-v1",
        "title": "Biohub Spotiflow Finetuned Acceptance v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-detection", "spotiflow", "validation"],
        "dataset_sources": [
            "indarkarhana/biohub-spotiflow-detector-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": ["indarkarhana/biohub-spotiflow-synthetic-finetune-v1"],
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
