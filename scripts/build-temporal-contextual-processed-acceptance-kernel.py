from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "temporal-contextual-pair-fusion-processed-acceptance-v3"
KERNEL_ID = "biohub-temporal-contextual-processed-acceptance-v3"
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

RUN_ID = "temporal-contextual-pair-fusion-processed-acceptance-v3"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/processed_launcher_terminal.json")
OUTPUT_DIR = Path("/kaggle/working/temporal_contextual_processed_acceptance_v3")
DECLARED_BUDGET_SECONDS = 21_600


def write_terminal(status, error=None):
    result = OUTPUT_DIR / "materialization_result.json"
    candidate = OUTPUT_DIR / "processed_candidate.csv"
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": DECLARED_BUDGET_SECONDS,
        "materializer_hard_stop_seconds": 19_800,
        "gpu_count_required": 2,
        "whole_movie_sharding_required": True,
        "materialization_result_exists": result.is_file(),
        "processed_candidate_exists": candidate.is_file(),
        "processed_ground_truth_read": False,
        "exact_processed_scoring_performed": False,
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    if error is not None:
        payload["error"] = str(error)
    for name, path in (("result", result), ("candidate", candidate)):
        if path.is_file():
            payload[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
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
print("Contextual processed materialization watchdog armed for 21,600 seconds.")
'''


SETUP = r'''import importlib
import shutil
import subprocess
import sys
import zipfile

import torch

EXPECTED_RUNTIME_MANIFEST_SHA256 = "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
EXPECTED_PROCESSED_CONTROL_SHA256 = "6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b"
EXPECTED_RAW_GRAPH_TREE_SHA256 = "559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e"
EXPECTED_PROCESSED_STEMS = {
    "44b6_12dfb391", "44b6_267148e4", "6bba_062c8d37", "6bba_07e24132"
}


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


def materialize_runtime_input(runtime_input, runtime):
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    for source in runtime_input.iterdir():
        if source.is_dir():
            shutil.copytree(source, runtime / source.name)
        elif source.is_file() and source.suffix != ".zip" and source.name != "dataset-metadata.json":
            shutil.copy2(source, runtime / source.name)
    for archive in runtime_input.glob("*.zip"):
        destination = runtime / archive.stem
        if destination.exists():
            raise RuntimeError(f"Ambiguous runtime directory and archive: {archive.name}")
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)


def unique_json_parent(filename, predicate):
    matches = []
    for path in Path("/kaggle/input").rglob(filename):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if predicate(payload, path.parent):
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError({"filename": filename, "matches": [str(path) for path in matches]})
    return matches[0]


def artifact_tree_sha256(root):
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise RuntimeError(f"Empty raw graph tree: {root}")
    value = bytearray()
    for path in files:
        value.extend(path.relative_to(root).as_posix().encode("utf-8"))
        value.extend(b"\0")
        value.extend(hashlib.sha256(path.read_bytes()).digest())
        value.extend(b"\0")
    return hashlib.sha256(bytes(value)).hexdigest()


if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"Contextual processed materialization requires exactly two GPUs, saw {torch.cuda.device_count()}"
    )
gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
print({"gpu_count": 2, "gpu_names": gpu_names})

input_root = Path("/kaggle/input")
runtime_input = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-temporal-contextual-transfer-runtime-v1"),
    Path("/kaggle/input/biohub-temporal-contextual-transfer-runtime-v1"),
])
competition = first_existing([
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
])
support = first_existing([
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
])
if runtime_input is None:
    runtime_input = next(
        (
            path.parent
            for path in input_root.rglob("SOURCE_MANIFEST.json")
            if hashlib.sha256(path.read_bytes()).hexdigest()
            == EXPECTED_RUNTIME_MANIFEST_SHA256
        ),
        None,
    )
if support is None:
    support = next(
        (path for path in input_root.iterdir() if any(path.rglob("*.whl"))), None
    )

appearance_output = unique_json_parent(
    "training_terminal.json",
    lambda payload, parent: (
        payload.get("run_id") == "temporal-contextual-pair-fusion-v3"
        and payload.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
        and payload.get("status") == "completed"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_trained") is True
        and payload.get("both_folds_improved") is True
        and (parent / "target_44b6" / "appearance_model.pt").is_file()
        and (parent / "target_6bba" / "appearance_model.pt").is_file()
    ),
)
trackastra_output = unique_json_parent(
    "training_terminal.json",
    lambda payload, parent: (
        payload.get("run_id") == "trackastra-dual-fold-synthetic-v1"
        and payload.get("status") == "completed"
        and payload.get("gpu_count") == 2
        and (parent / "target_44b6" / "model.pt").is_file()
        and (parent / "target_6bba" / "model.pt").is_file()
    ),
)
calibration_output = unique_json_parent(
    "calibration_terminal.json",
    lambda payload, _parent: (
        payload.get("run_id") == "temporal-contextual-pair-fusion-blend-v3"
        and payload.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
        and payload.get("status") == "completed"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and payload.get("processed_acceptance_ground_truth_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ),
)

processed_controls = [
    path for path in input_root.rglob("processed_validation.csv")
    if hashlib.sha256(path.read_bytes()).hexdigest()
    == EXPECTED_PROCESSED_CONTROL_SHA256
]
if len(processed_controls) != 1:
    raise RuntimeError({"processed_controls": [str(path) for path in processed_controls]})
processed_control = processed_controls[0]
raw_graph_roots = {
    path.parent
    for path in input_root.rglob("44b6_12dfb391.geff")
    if path.is_dir() and (path / "zarr.json").is_file()
}
raw_graph_roots = [
    path for path in raw_graph_roots
    if artifact_tree_sha256(path) == EXPECTED_RAW_GRAPH_TREE_SHA256
]
if len(raw_graph_roots) != 1:
    raise RuntimeError({"raw_graph_roots": [str(path) for path in raw_graph_roots]})
raw_graph_root = raw_graph_roots[0]
if None in (runtime_input, competition, support):
    raise FileNotFoundError({
        "runtime_input": runtime_input,
        "competition": competition,
        "support": support,
    })

runtime = Path("/kaggle/working/temporal_contextual_transfer_runtime_v1")
materialize_runtime_input(runtime_input, runtime)
runtime_manifest_sha256 = hashlib.sha256(
    (runtime / "SOURCE_MANIFEST.json").read_bytes()
).hexdigest()
if runtime_manifest_sha256 != EXPECTED_RUNTIME_MANIFEST_SHA256:
    raise RuntimeError(f"Contextual processed runtime changed: {runtime_manifest_sha256}")
subprocess.run(
    [sys.executable, str(runtime / "verify_runtime.py"), "--root", str(runtime), "--require-gpus"],
    check=True,
)

wheel_dirs = sorted({path.parent for path in support.rglob("*.whl")})
if not wheel_dirs:
    raise FileNotFoundError(f"No offline wheels found below {support}")
numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"Unexpected Kaggle NumPy before install: {numpy_before}")
pip_command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
for wheel_dir in wheel_dirs:
    pip_command.extend(["--find-links", str(wheel_dir)])
pip_command.extend([
    "bidict==0.23.1", "donfig==0.8.1.post1", "geff==1.2.0.1.1",
    "geff-spec==1.1.1", "ilpy==0.6.0", "numcodecs==0.15.1",
    "polars==1.42.0", "polars-runtime-32==1.42.0", "pyscipopt==6.2.1",
    "rustworkx==0.18.0", "tracksdata==0.1.0rc6.dev3+g980c2d30a", "zarr==3.2.1",
])
subprocess.run(pip_command, check=True)
for module in ("numpy", "scipy", "tracksdata", "zarr", "polars", "dask", "yaml"):
    importlib.import_module(module)
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("Offline install changed NumPy")

subprocess.run(
    [
        sys.executable, str(runtime / "verify_appearance_output.py"),
        "--root", str(appearance_output),
        "--expected-family", "temporal_contextual_pair_fusion_v3",
        "--strict-checkpoint",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, str(runtime / "verify_trackastra_output.py"),
        "--root", str(trackastra_output), "--allow-pretrained-control",
    ],
    check=True,
)
print(json.dumps({
    "runtime": str(runtime),
    "runtime_manifest_sha256": runtime_manifest_sha256,
    "competition": str(competition),
    "appearance_output": str(appearance_output),
    "calibration_terminal": str(calibration_output / "calibration_terminal.json"),
    "trackastra_output": str(trackastra_output),
    "processed_control": str(processed_control),
    "processed_control_sha256": EXPECTED_PROCESSED_CONTROL_SHA256,
    "raw_graph_root": str(raw_graph_root),
    "raw_graph_tree_sha256": EXPECTED_RAW_GRAPH_TREE_SHA256,
}, indent=2, sort_keys=True))
'''


MATERIALIZE = r'''command = [
    sys.executable,
    str(runtime / "dual_fold_appearance_processed_acceptance.py"),
    "--orchestrate",
    "--processed-control-csv", str(processed_control),
    "--raw-graph-root", str(raw_graph_root),
    "--competition-dir", str(competition),
    "--trackastra-output-root", str(trackastra_output),
    "--appearance-output-root", str(appearance_output),
    "--calibration-terminal", str(calibration_output / "calibration_terminal.json"),
    "--trackastra-dir", str(runtime / "trackastra_source"),
    "--output-dir", str(OUTPUT_DIR),
    "--max-tokens", "512",
    "--candidate-radius", "80.0",
    "--node-batch-size", "64",
    "--hard-stop-seconds", "19800",
]
print("Launching one-shot two-GPU contextual processed materialization:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as error:
    write_terminal("failed", error)
    raise

result_path = OUTPUT_DIR / "materialization_result.json"
candidate_path = OUTPUT_DIR / "processed_candidate.csv"
if not result_path.is_file() or not candidate_path.is_file():
    raise RuntimeError("Processed materializer exited without both evidence files")
result = json.loads(result_path.read_text(encoding="utf-8"))
if not (
    result.get("status") == "completed"
    and result.get("run_id") == RUN_ID
    and result.get("candidate_family") == "trackastra_contextual_pair_fusion_blend"
    and result.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
    and result.get("gpu_count") == 2
    and result.get("whole_movie_sharding") is True
    and result.get("processed_control_sha256") == EXPECTED_PROCESSED_CONTROL_SHA256
    and int(result.get("total_changed_edges", 0)) > 0
    and result.get("ground_truth_read") is False
    and result.get("public_leaderboard_used_for_selection") is False
    and result.get("hyperparameter_selection_performed") is False
    and result.get("exact_processed_scoring_performed") is False
    and result.get("competition_submission_performed") is False
    and result.get("authorized_for_submission") is False
):
    raise RuntimeError(f"Invalid contextual processed materialization: {result}")
print(json.dumps(result, indent=2, sort_keys=True))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("One-shot processed candidate materialized; exact scoring and submission did not run.")
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
                "# Contextual v3 one-shot processed materialization\n\n"
                "Exactly two GPUs apply already-frozen reciprocal contextual models and "
                "blend weights to four predeclared processed movies. Only images and "
                "hash-pinned comparator graphs are read. Ground truth, exact scoring, "
                "leaderboard selection, test inference, and submission are excluded.\n"
            ),
            code_cell(SETUP),
            code_cell(MATERIALIZE),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Temporal Contextual Processed Acceptance v3",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "acceptance", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-temporal-contextual-transfer-v3",
            "indarkarhana/biohub-temporal-contextual-calibration-v3",
            "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
            "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v2",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
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
