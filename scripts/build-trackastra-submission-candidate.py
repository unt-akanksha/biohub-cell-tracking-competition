from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-trackastra-submission-candidate-v1"
NOTEBOOK = TARGET / "biohub-trackastra-submission-candidate-v1.ipynb"


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

RUN_ID = "trackastra-submission-candidate-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    submission = Path("/kaggle/working/trackastra_candidate/submission.csv")
    report = Path("/kaggle/working/trackastra_candidate/candidate_report.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 7200,
        "hard_stop_seconds": 6900,
        "submission_exists": submission.is_file(),
        "report_exists": report.is_file(),
    }
    if error:
        payload["error"] = str(error)
    for name, path in (("submission", submission), ("report", report)):
        if path.is_file():
            payload[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
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
print("Trackastra submission watchdog armed for 6,900 seconds.")
'''


SETUP = r'''import shutil
import subprocess
import sys
import zipfile

BASE_SHA256 = "33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a"
RAW_GRAPH_TREE_SHA256 = "559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e"
input_root = Path("/kaggle/input")
runtime = next(
    (
        path
        for path in (
            Path("/kaggle/input/datasets/indarkarhana/biohub-trackastra-graph-runtime-v1"),
            Path("/kaggle/input/biohub-trackastra-graph-runtime-v1"),
        )
        if path.exists()
    ),
    None,
)
if runtime is None:
    runtime = next(
        (
            path
            for path in input_root.iterdir()
            if (path / "runtime_bundle.zip").is_file()
            or (path / "rerank_submission.py").is_file()
        ),
        None,
    )
if runtime is None:
    raise FileNotFoundError("Trackastra runtime input was not found")

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
manifest = json.loads((runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in ("trainer.py", "hybrid_linker.py", "rerank_submission.py"):
    actual = hashlib.sha256((runtime / name).read_bytes()).hexdigest()
    expected = manifest["files"][name]["sha256"]
    if actual != expected:
        raise RuntimeError(f"Runtime source hash mismatch for {name}")

base_candidates = [
    path
    for path in input_root.rglob("submission.csv")
    if hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA256
]
if len(base_candidates) != 1:
    raise FileNotFoundError(
        f"Expected one hash-pinned owned baseline submission, found {base_candidates}"
    )
base_submission = base_candidates[0]


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


raw_graph_roots = {
    path.parent
    for path in input_root.rglob("44b6_0113de3b.geff")
    if (path / "zarr.json").is_file()
}
raw_graph_roots = [
    path for path in raw_graph_roots if artifact_tree_sha256(path) == RAW_GRAPH_TREE_SHA256
]
if len(raw_graph_roots) != 1:
    raise FileNotFoundError(
        f"Expected one hash-pinned raw confidence graph root, found {raw_graph_roots}"
    )
raw_graph_root = raw_graph_roots[0]

training_terminals = []
for path in input_root.rglob("training_terminal.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if "association_acceptance_passed" in payload:
        training_terminals.append((path, payload))
if len(training_terminals) != 1:
    raise FileNotFoundError(
        f"Expected one learned association terminal, found {[str(p) for p, _ in training_terminals]}"
    )
training_terminal, training_payload = training_terminals[0]
model_dir = training_terminal.parent
model_path = model_dir / "model.pt"
if hashlib.sha256(model_path.read_bytes()).hexdigest() != training_payload["model_sha256"]:
    raise RuntimeError("Learned association model hash mismatch")
if not training_payload["association_acceptance_passed"]:
    print("Training-time acceptance did not pass; checking independent posthoc evidence.")

acceptance_terminals = []
for path in input_root.rglob("acceptance_terminal.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if payload.get("evaluation_kind") == "posthoc_complete_movie_acceptance":
        acceptance_terminals.append((path, payload))
if len(acceptance_terminals) != 1:
    raise FileNotFoundError(
        f"Expected one posthoc acceptance terminal, found {[str(p) for p, _ in acceptance_terminals]}"
    )
acceptance_terminal, acceptance_payload = acceptance_terminals[0]
if acceptance_payload.get("model_sha256") != training_payload["model_sha256"]:
    raise RuntimeError("Posthoc acceptance evidence is bound to another model")
if not acceptance_payload.get("association_acceptance_passed"):
    raise RuntimeError("Learned association candidate did not pass frozen posthoc acceptance")
if not __import__("torch").cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print(json.dumps({
    "runtime": str(runtime),
    "trackastra_dir": str(trackastra_dir),
    "base_submission": str(base_submission),
    "raw_graph_root": str(raw_graph_root),
    "model_dir": str(model_dir),
    "acceptance_terminal": str(acceptance_terminal),
    "base_sha256": BASE_SHA256,
    "model_sha256": training_payload["model_sha256"],
    "selected_method": acceptance_payload["complete_movie_selected"]["method"],
}, indent=2))
'''


INFER = r'''output_dir = Path("/kaggle/working/trackastra_candidate")
command = [
    sys.executable,
    str(runtime / "rerank_submission.py"),
    "--base-submission", str(base_submission),
    "--model-dir", str(model_dir),
    "--acceptance-terminal", str(acceptance_terminal),
    "--trackastra-dir", str(trackastra_dir),
    "--base-graph-root", str(raw_graph_root),
    "--output-dir", str(output_dir),
    "--max-tokens", "512",
    "--candidate-radius", "80",
]
print("Launching learned association inference:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as exc:
    write_terminal("failed", exc)
    raise

submission = output_dir / "submission.csv"
report = output_dir / "candidate_report.json"
if not submission.is_file() or not report.is_file():
    raise RuntimeError("Reranker exited without a submission and report")
summary = json.loads(report.read_text(encoding="utf-8"))
if summary["candidate_submission_sha256"] == BASE_SHA256:
    raise RuntimeError("Candidate is an exact public baseline replica")
if summary["total_changed_edges"] <= 0:
    raise RuntimeError("Candidate did not change any association edges")
shutil.copy2(submission, "/kaggle/working/submission.csv")
print(json.dumps({
    "candidate_sha256": summary["candidate_submission_sha256"],
    "base_sha256": summary["base_submission_sha256"],
    "association_method": summary["association_method"],
    "clean_acceptance_delta": summary["clean_acceptance_delta_vs_base_raw"],
    "total_changed_edges": summary["total_changed_edges"],
}, indent=2))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Independent learned-association submission candidate is ready.")
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
                "# Learned Trackastra submission candidate\n\n"
                "This candidate preserves the clean baseline detector coordinates and replaces "
                "associations with a Biohub-fine-tuned Trackastra model using hash-pinned raw "
                "edge confidence. It can run only after a frozen heldout acceptance improvement "
                "and refuses an exact replica.\n"
            ),
            code_cell(SETUP),
            code_cell(INFER),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    metadata = {
        "id": "indarkarhana/biohub-trackastra-submission-candidate-v1",
        "title": "Biohub Trackastra Submission Candidate v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "trackastra"],
        "dataset_sources": ["indarkarhana/biohub-trackastra-graph-runtime-v1"],
        "kernel_sources": [
            "indarkarhana/biohub-clean-0-927-reproduction-v1",
            "indarkarhana/biohub-trackastra-graph-finetune-v6",
            "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v1",
        ],
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
