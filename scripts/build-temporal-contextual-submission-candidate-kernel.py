from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "temporal-contextual-pair-fusion-candidate-v3"
KERNEL_ID = "biohub-temporal-contextual-submission-candidate-v3"
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

RUN_ID = "temporal-contextual-pair-fusion-candidate-v3"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/candidate_launcher_terminal.json")
OUTPUT_DIR = Path("/kaggle/working/temporal_contextual_candidate_v3")
DECLARED_BUDGET_SECONDS = 43_200
INFERENCE_HARD_STOP_SECONDS = 36_000
FINALIZATION_RESERVE_SECONDS = 7_200


def write_terminal(status, error=None):
    candidate = OUTPUT_DIR / "submission.csv"
    report = OUTPUT_DIR / "candidate_report.json"
    root_candidate = Path("/kaggle/working/submission.csv")
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": DECLARED_BUDGET_SECONDS,
        "inference_hard_stop_seconds": INFERENCE_HARD_STOP_SECONDS,
        "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "gpu_count_required": 2,
        "whole_movie_sharding_required": True,
        "candidate_exists": candidate.is_file(),
        "candidate_report_exists": report.is_file(),
        "root_candidate_exists": root_candidate.is_file(),
        "public_leaderboard_used_for_selection": False,
        "competition_submission_performed": False,
        "authorized_for_submission": False,
    }
    if error is not None:
        payload["error"] = str(error)
    for name, path in (("candidate", candidate), ("report", report)):
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
print("Contextual submission-candidate watchdog armed for 43,200 seconds.")
'''


SETUP = r'''import importlib
import shutil
import subprocess
import sys
import zipfile

import torch

EXPECTED_RUNTIME_MANIFEST_SHA256 = "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
EXPECTED_BASE_SHA256 = "33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a"
EXPECTED_RAW_GRAPH_TREE_SHA256 = "559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e"


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
        f"Contextual candidate construction requires exactly two GPUs, saw {torch.cuda.device_count()}"
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

acceptance_candidates = []
for path in input_root.rglob("*.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if (
        payload.get("status") == "accepted"
        and payload.get("evaluation_kind") == "exact_processed_dual_fold_acceptance"
        and payload.get("exact_processed_gate_passed") is True
        and payload.get("candidate_family") == "trackastra_contextual_pair_fusion_blend"
        and payload.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("competition_submission_performed") is False
    ):
        acceptance_candidates.append(path)
if len(acceptance_candidates) != 1:
    raise RuntimeError({"accepted_evidence": [str(path) for path in acceptance_candidates]})
acceptance_evidence = acceptance_candidates[0]

base_candidates = [
    path for path in input_root.rglob("submission.csv")
    if hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_BASE_SHA256
]
if len(base_candidates) != 1:
    raise RuntimeError({"base_candidates": [str(path) for path in base_candidates]})
base_submission = base_candidates[0]
raw_graph_roots = {
    path.parent
    for path in input_root.rglob("44b6_0113de3b.geff")
    if path.is_dir() and (path / "zarr.json").is_file()
}
raw_graph_roots = [
    path for path in raw_graph_roots
    if artifact_tree_sha256(path) == EXPECTED_RAW_GRAPH_TREE_SHA256
]
if len(raw_graph_roots) != 1:
    raise RuntimeError({"raw_graph_roots": [str(path) for path in raw_graph_roots]})
base_graph_root = raw_graph_roots[0]
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
    raise RuntimeError(f"Contextual candidate runtime changed: {runtime_manifest_sha256}")
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
    "competition_test": str(competition / "test"),
    "appearance_output": str(appearance_output),
    "trackastra_output": str(trackastra_output),
    "acceptance_evidence": str(acceptance_evidence),
    "acceptance_evidence_sha256": hashlib.sha256(acceptance_evidence.read_bytes()).hexdigest(),
    "base_submission": str(base_submission),
    "base_submission_sha256": EXPECTED_BASE_SHA256,
    "base_graph_root": str(base_graph_root),
    "base_graph_tree_sha256": EXPECTED_RAW_GRAPH_TREE_SHA256,
}, indent=2, sort_keys=True))
'''


INFER = r'''command = [
    sys.executable,
    str(runtime / "dual_fold_appearance_submission.py"),
    "--orchestrate",
    "--base-submission", str(base_submission),
    "--base-graph-root", str(base_graph_root),
    "--image-root", str(competition / "test"),
    "--trackastra-44b6-dir", str(trackastra_output / "target_44b6"),
    "--trackastra-6bba-dir", str(trackastra_output / "target_6bba"),
    "--appearance-44b6-model", str(appearance_output / "target_44b6" / "appearance_model.pt"),
    "--appearance-6bba-model", str(appearance_output / "target_6bba" / "appearance_model.pt"),
    "--acceptance-evidence", str(acceptance_evidence),
    "--trackastra-dir", str(runtime / "trackastra_source"),
    "--output-dir", str(OUTPUT_DIR),
    "--max-tokens", "512",
    "--candidate-radius", "80.0",
    "--node-batch-size", "64",
    "--hard-stop-seconds", "36000",
]
print("Launching two-GPU whole-movie contextual candidate inference:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as error:
    write_terminal("failed", error)
    raise

candidate = OUTPUT_DIR / "submission.csv"
report_path = OUTPUT_DIR / "candidate_report.json"
if not candidate.is_file() or not report_path.is_file():
    raise RuntimeError("Contextual inference exited without candidate and report")
report = json.loads(report_path.read_text(encoding="utf-8"))
if not (
    report.get("status") == "completed"
    and report.get("candidate_family") == "trackastra_contextual_pair_fusion_blend"
    and report.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
    and report.get("gpu_count") == 2
    and report.get("inference_hard_stop_seconds") == INFERENCE_HARD_STOP_SECONDS
    and report.get("notebook_runtime_reserve_seconds") >= FINALIZATION_RESERVE_SECONDS
    and report.get("base_submission_sha256") == EXPECTED_BASE_SHA256
    and report.get("candidate_submission_sha256") != EXPECTED_BASE_SHA256
    and int(report.get("total_changed_edges", 0)) > 0
    and len(report.get("whole_movie_coverage", [])) > 0
    and len(report.get("whole_movie_coverage", []))
        == len(set(report.get("whole_movie_coverage", [])))
    and report.get("nodes_preserved_exactly") is True
    and report.get("public_leaderboard_used_for_selection") is False
    and report.get("competition_submission_performed") is False
):
    raise RuntimeError(f"Invalid contextual candidate report: {report}")
shutil.copy2(candidate, "/kaggle/working/submission.csv")
print(json.dumps(report, indent=2, sort_keys=True))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Accepted contextual candidate is ready locally; no Kaggle submission was performed.")
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
                "# Accepted contextual v3 submission candidate\n\n"
                "Exactly two T4 GPUs balance complete movies using contextual pair and "
                "patch workload. Hash-bound exact processed acceptance is mandatory. "
                "The notebook reserves two hours for finalization, refuses an edge-identical "
                "public replica, writes a local CSV, and contains no upload command.\n"
            ),
            code_cell(SETUP),
            code_cell(INFER),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Temporal Contextual Submission Candidate v3",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "submission-candidate", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
            "indarkarhana/biohub-temporal-contextual-exact-acceptance-v3",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-clean-0-927-reproduction-v1",
            "indarkarhana/biohub-temporal-contextual-transfer-v3",
            "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
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
