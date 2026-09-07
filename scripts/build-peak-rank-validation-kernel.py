from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-peak-rank-validation-v1"
NOTEBOOK = TARGET / "biohub-peak-rank-validation-v1.ipynb"


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

RUN_ID = "peak-rank-validation-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    result = Path("/kaggle/working/peak_rank_validation/peak_rank_validation.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 43200,
        "hard_stop_seconds": 42000,
        "validation_result_exists": result.is_file(),
        "competition_submission_performed": False,
    }
    if error:
        payload["error"] = str(error)
    if result.is_file():
        payload["validation_result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
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
TIMER = threading.Timer(42000, budget_expired)
TIMER.daemon = True
TIMER.start()
print("Two-GPU peak-ranking validation watchdog armed for 42,000 seconds.")
'''


SETUP = r'''import importlib
import subprocess
import sys


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


input_root = Path("/kaggle/input")
runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-peak-rank-validation-runtime-v1"),
    Path("/kaggle/input/biohub-peak-rank-validation-runtime-v1"),
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
if runtime is None:
    runtime = next((p for p in input_root.iterdir() if (p / "evaluate_peak_rank_detector.py").is_file()), None)
if graph_runtime is None:
    graph_runtime = next((p for p in input_root.iterdir() if (p / "validator_raw/44b6_12dfb391.geff/zarr.json").is_file()), None)
if support is None:
    support = next((p for p in input_root.iterdir() if any(p.rglob("tracksdata*.whl"))), None)
if None in (runtime, graph_runtime, competition, support):
    raise FileNotFoundError({
        "runtime": runtime,
        "graph_runtime": graph_runtime,
        "competition": competition,
        "support": support,
    })

wheel_dirs = sorted({p.parent for p in support.rglob("*.whl")})
if not wheel_dirs:
    raise FileNotFoundError("support pack contains no offline wheels")
install = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
for wheel_dir in wheel_dirs:
    install.extend(["--find-links", str(wheel_dir)])
install.extend([
    "bidict==0.23.1", "donfig==0.8.1.post1", "geff==1.2.0.1.1",
    "geff-spec==1.1.1", "numcodecs==0.15.1", "polars==1.42.0",
    "polars-runtime-32==1.42.0", "rustworkx==0.18.0",
    "tracksdata==0.1.0rc6.dev3+g980c2d30a", "zarr==3.2.1",
])
subprocess.run(install, check=True)
support_repo = next((p for p in support.rglob("biohub_tracking") if p.is_dir()), None)
if support_repo is None:
    raise FileNotFoundError("biohub_tracking source not found in support pack")
sys.path.insert(0, str(runtime))
sys.path.insert(0, str(support_repo.parent))
for module in ("numpy", "scipy", "torch", "tracksdata", "zarr"):
    imported = importlib.import_module(module)
    print(module, getattr(imported, "__version__", "unknown"))
torch = importlib.import_module("torch")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(f"exactly two CUDA GPUs required, found {torch.cuda.device_count()}")

manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
if manifest.get("training_audit_passed") is not True or manifest.get("parameter_count") != 38381478:
    raise RuntimeError("runtime manifest does not contain an accepted detector")
for name, row in manifest["files"].items():
    path = runtime / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError(f"runtime file hash mismatch: {name}")
probe = importlib.import_module("tracksdata").graph.IndexedRXGraph.from_geff(
    graph_runtime / "validator_raw" / "44b6_12dfb391.geff"
)
probe = probe[0] if isinstance(probe, tuple) else probe
if probe.node_attrs().height <= 0:
    raise RuntimeError("baseline graph IO probe is empty")
print(json.dumps({
    "runtime": str(runtime),
    "competition": str(competition),
    "graph_runtime": str(graph_runtime),
    "gpu_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(2)],
}, indent=2))
'''


VALIDATE = r'''output_dir = Path("/kaggle/working/peak_rank_validation")
command = [
    sys.executable,
    str(runtime / "evaluate_peak_rank_detector.py"),
    "--competition-dir", str(competition),
    "--checkpoint", str(runtime / "peak_rank_detector.pt"),
    "--training-terminal", str(runtime / "training_terminal.json"),
    "--baseline-predictions", str(graph_runtime / "validator_raw"),
    "--output-dir", str(output_dir),
    "--devices", "0,1",
    "--batch-size", "1",
    "--calibration-frames", "12",
    "--max-wall-seconds", "39000",
]
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join(
    [str(runtime), str(support_repo.parent), environment.get("PYTHONPATH", "")]
).rstrip(os.pathsep)
print("Launching two-GPU clean peak-ranking validation:", " ".join(command))
try:
    subprocess.run(command, check=True, env=environment)
except Exception as exc:
    write_terminal("validation_failed", exc)
    raise
result_path = output_dir / "peak_rank_validation.json"
if not result_path.is_file():
    raise RuntimeError("evaluator exited without a validation result")
result = json.loads(result_path.read_text(encoding="utf-8"))
print(json.dumps({
    "selection_passed": result["selection_passed"],
    "acceptance_opened": result["acceptance_opened"],
    "promotion_passed": result["promotion_passed"],
    "selection_recall": result["selection"]["annotated_node_recall"],
    "acceptance_recall": None if result["acceptance"] is None else result["acceptance"]["annotated_node_recall"],
}, indent=2, sort_keys=True))
'''


FINISH = r'''unexpected = [
    path for path in Path("/kaggle/working").rglob("*")
    if path.is_file() and path.name.lower() in {"submission.csv", "submission.zip"}
]
if unexpected:
    raise RuntimeError(f"validation unexpectedly created submission artifacts: {unexpected}")
FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Peak-ranking clean validation complete; no submission was created.")
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
                "# Independent temporal peak-ranking detector validation\n\n"
                "Two GPUs score frozen held-out movies. The acceptance set remains closed "
                "unless the separate selection set passes; this notebook creates no submission.\n"
            ),
            code_cell(SETUP),
            code_cell(VALIDATE),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": "indarkarhana/biohub-peak-rank-validation-v1",
        "title": "Biohub Peak Rank Validation v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "3d-unet", "peak-ranking", "validation"],
        "dataset_sources": [
            "indarkarhana/biohub-peak-rank-validation-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        # Kaggle's metadata name for the dual-T4 accelerator is NvidiaTeslaT4;
        # the notebook additionally refuses to run unless two devices exist.
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
