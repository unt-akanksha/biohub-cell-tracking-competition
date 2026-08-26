from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle" / "biohub-hoct-submission-candidate-v1"
NOTEBOOK = TARGET / "biohub-hoct-submission-candidate-v1.ipynb"


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

RUN_ID = "hoct-submission-candidate-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    submission = Path("/kaggle/working/submission.csv")
    report = Path("/kaggle/working/hoct_candidate_report.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 10800,
        "hard_stop_seconds": 10500,
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
TIMER = threading.Timer(10500, budget_expired)
TIMER.daemon = True
TIMER.start()
print("HOCT candidate watchdog armed for 10,500 seconds.")
'''


SETUP = r'''import importlib
import shutil
import subprocess
import sys
import zipfile

BASE_SHA256 = "33c179b0449b9cdd186f06a653cddc8cf12359f008982f6713cdf30784a52e6a"
RAW_GRAPH_TREE_SHA256 = "559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e"
RERANKER_SHA256 = "4bbd74c32935faefa17870255d2f1f68f66e83a157bcee935fddedfe2bdddd36"
input_root = Path("/kaggle/input")
hoct_runtime = next((path for path in (
    Path("/kaggle/input/datasets/indarkarhana/biohub-hoct-candidate-runtime-v1"),
    Path("/kaggle/input/biohub-hoct-candidate-runtime-v1"),
) if path.exists()), None)
graph_runtime = next((path for path in (
    Path("/kaggle/input/datasets/indarkarhana/biohub-trackastra-graph-runtime-v1"),
    Path("/kaggle/input/biohub-trackastra-graph-runtime-v1"),
) if path.exists()), None)
support = next((path for path in (
    Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
    Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
) if path.exists()), None)
if None in (hoct_runtime, graph_runtime, support):
    raise FileNotFoundError({
        "hoct_runtime": hoct_runtime,
        "graph_runtime": graph_runtime,
        "support": support,
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
hoct_manifest = json.loads((hoct_runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in (
    "association_ensemble.py", "biohub_adapter.py", "multibackbone.py",
    "train_biohub_hoct_probe.py", "rerank_hoct_multibackbone_submission.py",
    "general_v1.pt", "ctc_v0.pt",
):
    actual = hashlib.sha256((hoct_runtime / name).read_bytes()).hexdigest()
    if actual != hoct_manifest["files"][name]["sha256"]:
        raise RuntimeError(f"HOCT candidate runtime hash mismatch: {name}")
if hoct_manifest["files"]["rerank_hoct_multibackbone_submission.py"]["sha256"] != RERANKER_SHA256:
    raise RuntimeError("HOCT candidate reranker binding mismatch")
graph_manifest = json.loads((graph_runtime / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for name in ("trainer.py", "hybrid_linker.py", "rerank_submission.py"):
    actual = hashlib.sha256((graph_runtime / name).read_bytes()).hexdigest()
    if actual != graph_manifest["files"][name]["sha256"]:
        raise RuntimeError(f"Graph runtime hash mismatch: {name}")

wheel_dirs = sorted({path.parent for path in support.rglob("*.whl")})
numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2" or not wheel_dirs:
    raise RuntimeError({"numpy": numpy_before, "wheel_dirs": wheel_dirs})
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
for module in ("numpy", "scipy", "torch", "tracksdata", "zarr", "polars"):
    importlib.import_module(module)
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("Offline dependency install changed NumPy")
if not importlib.import_module("torch").cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

base_candidates = [
    path for path in input_root.rglob("submission.csv")
    if hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA256
]
if len(base_candidates) != 1:
    raise FileNotFoundError(f"Expected one hash-pinned owned base CSV, found {base_candidates}")
base_submission = base_candidates[0]


def artifact_tree_sha256(root):
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    value = bytearray()
    for path in files:
        value.extend(path.relative_to(root).as_posix().encode())
        value.extend(b"\0")
        value.extend(hashlib.sha256(path.read_bytes()).digest())
        value.extend(b"\0")
    return hashlib.sha256(bytes(value)).hexdigest()


raw_graph_roots = {
    path.parent for path in input_root.rglob("44b6_0113de3b.geff")
    if (path / "zarr.json").is_file()
}
raw_graph_roots = [
    path for path in raw_graph_roots
    if artifact_tree_sha256(path) == RAW_GRAPH_TREE_SHA256
]
if len(raw_graph_roots) != 1:
    raise FileNotFoundError(f"Expected one hash-pinned raw graph root, found {raw_graph_roots}")
raw_graph_root = raw_graph_roots[0]

accepted = []
for terminal_path in input_root.rglob("training_terminal.json"):
    try:
        payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    probe_path = terminal_path.parent / "hoct_multibackbone_probes.pt"
    if (
        payload.get("status") == "completed"
        and payload.get("association_acceptance_passed") is True
        and payload.get("probe_checkpoint_sha256")
        and probe_path.is_file()
        and hashlib.sha256(probe_path.read_bytes()).hexdigest() == payload["probe_checkpoint_sha256"]
    ):
        accepted.append((terminal_path, probe_path, payload))
if len(accepted) != 1:
    raise FileNotFoundError(f"Expected one clean-accepted HOCT probe, found {[str(row[0]) for row in accepted]}")
acceptance_terminal, probe_path, acceptance = accepted[0]
print(json.dumps({
    "base_submission": str(base_submission),
    "raw_graph_root": str(raw_graph_root),
    "probe": str(probe_path),
    "acceptance_terminal": str(acceptance_terminal),
    "selected": acceptance["selected"],
}, indent=2))
'''


INFER = r'''command = [
    sys.executable,
    str(hoct_runtime / "rerank_hoct_multibackbone_submission.py"),
    "--base-submission", str(base_submission),
    "--raw-graph-root", str(raw_graph_root),
    "--general-model", str(hoct_runtime / "general_v1.pt"),
    "--ctc-model", str(hoct_runtime / "ctc_v0.pt"),
    "--probes", str(probe_path),
    "--acceptance-terminal", str(acceptance_terminal),
    "--output", "/kaggle/working/submission.csv",
    "--report", "/kaggle/working/hoct_candidate_report.json",
    "--core-size", "128",
]
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join([str(hoct_runtime), str(graph_runtime)])
print("Launching accepted HOCT candidate inference:", " ".join(command))
try:
    subprocess.run(command, check=True, env=environment)
except Exception as exc:
    write_terminal("failed", exc)
    raise
report = json.loads(Path("/kaggle/working/hoct_candidate_report.json").read_text(encoding="utf-8"))
if report["base_submission_sha256"] != BASE_SHA256 or report["total_edge_changes"] <= 0:
    raise RuntimeError("HOCT candidate failed public-replica refusal")
print(json.dumps({
    "candidate_sha256": report["output_submission_sha256"],
    "base_sha256": report["base_submission_sha256"],
    "total_edge_changes": report["total_edge_changes"],
    "selected": report["selected"],
}, indent=2))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Clean-accepted independent HOCT submission candidate is ready.")
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
                "# Clean-accepted HOCT submission candidate\n\n"
                "This notebook preserves the frozen detector nodes and uses independent "
                "complementary HOCT backbones plus Biohub-supervised probes. It can run only "
                "after disjoint complete-movie acceptance improves and refuses an edge-identical "
                "public replica.\n"
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
        "id": "indarkarhana/biohub-hoct-submission-candidate-v1",
        "title": "Biohub HOCT Submission Candidate v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "transformer"],
        "dataset_sources": [
            "indarkarhana/biohub-hoct-candidate-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-clean-0-927-reproduction-v1",
            "indarkarhana/biohub-hoct-multibackbone-probe-v1",
        ],
        "competition_sources": [],
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
