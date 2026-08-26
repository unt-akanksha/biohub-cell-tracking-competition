from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-spotiflow-pu-adaptation-v1"
NOTEBOOK = TARGET / "biohub-spotiflow-pu-adaptation-v1.ipynb"


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

RUN_ID = "spotiflow-pu-adaptation-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    result = Path("/kaggle/working/spotiflow_pu_validation/pu_detector_validation.json")
    training = Path("/kaggle/working/spotiflow_pu_model/pu_training_result.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 7200,
        "hard_stop_seconds": 6900,
        "training_result_exists": training.is_file(),
        "validation_result_exists": result.is_file(),
        "competition_submission_performed": False,
    }
    if error:
        payload["error"] = str(error)
    if training.is_file():
        payload["training_result_sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
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
TIMER = threading.Timer(6900, budget_expired)
TIMER.daemon = True
TIMER.start()
print("Spotiflow PU watchdog armed for 6,900 seconds.")
'''


SETUP = r'''import importlib
import subprocess
import sys


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


input_root = Path("/kaggle/input")
runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-spotiflow-pu-runtime-v1"),
    Path("/kaggle/input/biohub-spotiflow-pu-runtime-v1"),
])
graph_runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-trackastra-graph-runtime-v1"),
    Path("/kaggle/input/biohub-trackastra-graph-runtime-v1"),
])
competition = first_existing([
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
])
primary_support = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
])
secondary_support = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1"),
    Path("/kaggle/input/biohub-temporal-unet3d-seed314159-v1"),
])
if runtime is None:
    runtime = next((p for p in input_root.iterdir() if (p / "train_pu_detector.py").is_file()), None)
if graph_runtime is None:
    graph_runtime = next((p for p in input_root.iterdir() if (p / "validator_raw/44b6_12dfb391.geff/zarr.json").is_file()), None)
if primary_support is None:
    primary_support = next((p for p in input_root.iterdir() if (p / "weights/unet_transformer/split_0/edge_predictor_best.pth").is_file()), None)
if secondary_support is None:
    secondary_support = next((p for p in input_root.iterdir() if (p / "weights/unet_transformer/split_0/SNAPSHOT_MANIFEST.json").is_file()), None)
if None in (runtime, graph_runtime, competition, primary_support, secondary_support):
    raise FileNotFoundError({
        "runtime": runtime,
        "graph_runtime": graph_runtime,
        "competition": competition,
        "primary_support": primary_support,
        "secondary_support": secondary_support,
    })

numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"unexpected Kaggle NumPy before offline install: {numpy_before}")
support_wheel_dirs = sorted({p.parent for p in primary_support.rglob("*.whl")})
if not support_wheel_dirs:
    raise FileNotFoundError("primary support pack contains no dependency wheels")
graph_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
for wheel_dir in support_wheel_dirs:
    graph_cmd.extend(["--find-links", str(wheel_dir)])
graph_cmd.extend([
    "bidict==0.23.1", "donfig==0.8.1.post1", "geff==1.2.0.1.1",
    "geff-spec==1.1.1", "ilpy==0.6.0", "numcodecs==0.15.1",
    "polars==1.42.0", "polars-runtime-32==1.42.0", "pyscipopt==6.2.1",
    "rustworkx==0.18.0", "tracksdata==0.1.0rc6.dev3+g980c2d30a", "zarr==3.2.1",
])
subprocess.run(graph_cmd, check=True)
runtime_wheels = sorted((runtime / "wheels").glob("*.whl"))
if not runtime_wheels:
    raise FileNotFoundError("PU runtime contains no Spotiflow wheels")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, runtime_wheels)],
    check=True,
)

support_repo = next((p for p in primary_support.rglob("biohub_tracking") if p.is_dir()), None)
if support_repo is None:
    raise FileNotFoundError("biohub_tracking source not found in primary support")
sys.path.insert(0, str(support_repo.parent))
sys.path.insert(0, str(runtime))
for module in ("numpy", "scipy", "torch", "tracksdata", "zarr", "spotiflow"):
    imported = importlib.import_module(module)
    print(module, getattr(imported, "__version__", "unknown"))
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("offline dependency installation changed NumPy")
torch = importlib.import_module("torch")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in (
    "pu_targets.py", "public_teacher.py", "train_pu_detector.py",
    "evaluate_pu_detector.py", "evaluate_pretrained_detector.py", "density_calibration.py",
):
    path = runtime / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["scripts"][name]:
        raise RuntimeError(f"runtime script hash mismatch: {name}")

primary_weight = primary_support / "weights/unet_transformer/split_0/edge_predictor_best.pth"
secondary_weight = secondary_support / "weights/unet_transformer/split_0/edge_predictor_best.pth"
expected_weights = {
    str(primary_weight): "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    str(secondary_weight): "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
}
for name, expected in expected_weights.items():
    path = Path(name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"teacher hash mismatch: {path} {actual}")

probe = importlib.import_module("tracksdata").graph.IndexedRXGraph.from_geff(
    graph_runtime / "validator_raw" / "44b6_12dfb391.geff"
)
probe = probe[0] if isinstance(probe, tuple) else probe
if probe.node_attrs().height <= 0 or probe.edge_attrs().height <= 0:
    raise RuntimeError("baseline graph IO probe is empty")
print(json.dumps({
    "runtime": str(runtime),
    "competition": str(competition),
    "graph_runtime": str(graph_runtime),
    "primary_weight": str(primary_weight),
    "secondary_weight": str(secondary_weight),
    "cuda_device": torch.cuda.get_device_name(0),
}, indent=2))
'''


TRAIN = r'''model_dir = Path("/kaggle/working/spotiflow_pu_model")
train_command = [
    sys.executable,
    str(runtime / "train_pu_detector.py"),
    "--competition-dir", str(competition),
    "--spotiflow-model", str(runtime / "models/smfish_3d"),
    "--primary-teacher", str(primary_weight),
    "--secondary-teacher", str(secondary_weight),
    "--output-dir", str(model_dir),
    "--steps", "1024",
    "--min-steps", "256",
    "--warmup-steps", "256",
    "--pairs-per-movie", "3",
    "--learning-rate", "1e-5",
    "--seed", "20260826",
    "--max-wall-seconds", "5200",
]
print("Launching PU adaptation:", " ".join(train_command))
run_env = os.environ.copy()
run_env["PYTHONPATH"] = os.pathsep.join(
    [str(runtime), str(support_repo.parent), run_env.get("PYTHONPATH", "")]
).rstrip(os.pathsep)
try:
    subprocess.run(train_command, check=True, env=run_env)
except Exception as exc:
    write_terminal("training_failed", exc)
    raise
training_result = model_dir / "pu_training_result.json"
if not training_result.is_file() or not (model_dir / "best.pt").is_file():
    raise RuntimeError("PU trainer exited without learned checkpoint evidence")
print(json.dumps(json.loads(training_result.read_text(encoding="utf-8")), indent=2, sort_keys=True))
'''


VALIDATE = r'''validation_dir = Path("/kaggle/working/spotiflow_pu_validation")
validation_command = [
    sys.executable,
    str(runtime / "evaluate_pu_detector.py"),
    "--competition-dir", str(competition),
    "--model-dir", str(model_dir),
    "--training-result", str(training_result),
    "--baseline-predictions", str(graph_runtime / "validator_raw"),
    "--output-dir", str(validation_dir),
    "--calibration-frames", "12",
    "--max-wall-seconds", "1300",
]
print("Launching frozen PU validation:", " ".join(validation_command))
try:
    subprocess.run(validation_command, check=True, env=run_env)
except Exception as exc:
    write_terminal("validation_failed", exc)
    raise
validation_result = validation_dir / "pu_detector_validation.json"
if not validation_result.is_file():
    raise RuntimeError("PU evaluator exited without terminal validation evidence")
summary = json.loads(validation_result.read_text(encoding="utf-8"))
print(json.dumps({
    "selection_passed": summary["selection_passed"],
    "acceptance_opened": summary["acceptance_opened"],
    "promotion_passed": summary["promotion_passed"],
    "selection_recall": summary["selection"]["annotated_node_recall"],
    "acceptance_recall": None if summary["acceptance"] is None else summary["acceptance"]["annotated_node_recall"],
}, indent=2, sort_keys=True))
'''


FINISH = r'''unexpected_submissions = [
    path for path in Path("/kaggle/working").rglob("*")
    if path.is_file() and path.name.lower() in {"submission.csv", "submission.zip"}
]
if unexpected_submissions:
    raise RuntimeError(f"Validation unexpectedly created submission artifacts: {unexpected_submissions}")
FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Spotiflow PU training and clean validation complete; no submission was created.")
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
                "# Biohub positive-unlabeled Spotiflow adaptation\n\n"
                "Independent 35.5M-parameter detector adaptation using two frozen public teachers "
                "only for conservative pseudo-labels. Twelve validation movies are excluded from "
                "training; selection precedes untouched acceptance; no submission is created.\n"
            ),
            code_cell(SETUP),
            code_cell(TRAIN),
            code_cell(VALIDATE),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": "indarkarhana/biohub-spotiflow-pu-adaptation-v1",
        "title": "Biohub Spotiflow PU Adaptation v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-detection", "spotiflow", "positive-unlabeled"],
        "dataset_sources": [
            "indarkarhana/biohub-spotiflow-pu-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
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
