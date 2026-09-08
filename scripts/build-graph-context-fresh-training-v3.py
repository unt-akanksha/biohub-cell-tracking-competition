from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ID = "biohub-graph-context-fresh-training-runtime-v3"
KERNEL_ID = "biohub-graph-context-fresh-training-v3"
STAGING = ROOT / ".biohub" / "staging" / RUNTIME_ID
KERNEL_ROOT = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = KERNEL_ROOT / f"{KERNEL_ID}.ipynb"
ARCHIVE = (
    ROOT
    / ".biohub/cache/graph-context-relational-v1"
    / "biohub_graph_context_relational_patches_v1.tar.gz"
)
STAGED_ARCHIVE_NAME = "graph_context_relational_patches_v1.tar.bin"
PRETRAIN_ROOT = (
    ROOT
    / ".biohub/cache/kernel-outputs"
    / "zebrahub-multiscale-pretrain-v1-version1-error-20260829"
    / "zebrahub_multiscale_contextual_pretrain_v1"
)
SPLIT = ROOT / "research/graph_context_fresh_split_v3.json"
EXPECTED = {
    "archive": "efa3b5af80c75b2bc091ecda1e86064660dcfffe197e2a4950a12981248b0c3e",
    "split": "8d53a55217be88efc895ae9bf0bf378a3a4acb6c42437836a342d888cc9296da",
    "target_44b6": "a0a1794134893d191d896b8b83abee61a754bc3852e2b8f74dacd353ce118a76",
    "target_6bba": "9e8af9aeb247297d3ed09bda3bcc2b5af413d78c6bf2546eff5f4898667e074d",
}
SOURCE_FILES = (
    "research/train_handcrafted_division_gate.py",
    "research/temporal_contrastive/__init__.py",
    "research/temporal_contrastive/graph_context_division_model.py",
    "research/temporal_contrastive/multiscale_contextual_pair_fusion.py",
    "research/temporal_contrastive/relational_division_model.py",
    "research/temporal_contrastive/train_focused_division_gate.py",
    "research/temporal_contrastive/train_graph_context_division_sweep.py",
    "research/temporal_contrastive/train_graph_context_fresh_ensemble_v3.py",
    "research/temporal_contrastive/train_real_division_gate.py",
    "research/temporal_contrastive/train_relational_division_sweep.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def build_runtime() -> dict:
    if STAGING.exists():
        resolved = STAGING.resolve()
        if ROOT.resolve() not in resolved.parents or resolved.name != RUNTIME_ID:
            raise RuntimeError(f"refusing to replace unexpected path: {resolved}")
        shutil.rmtree(resolved)
    STAGING.mkdir(parents=True)
    inputs = {
        "archive": ARCHIVE,
        "split": SPLIT,
        "target_44b6": PRETRAIN_ROOT / "target_44b6/pretrained_model.pt",
        "target_6bba": PRETRAIN_ROOT / "target_6bba/pretrained_model.pt",
    }
    for name, path in inputs.items():
        if sha256_file(path) != EXPECTED[name]:
            raise RuntimeError(f"frozen input changed: {name}")
    # Kaggle expands recognized archive suffixes during dataset ingestion. Keep
    # the gzip-compressed tar bytes opaque so the notebook can verify and unpack
    # the exact frozen artifact itself.
    shutil.copy2(ARCHIVE, STAGING / STAGED_ARCHIVE_NAME)
    shutil.copy2(SPLIT, STAGING / SPLIT.name)
    shutil.copy2(PRETRAIN_ROOT / "pretraining_terminal.json", STAGING)
    for fold in ("target_44b6", "target_6bba"):
        shutil.copy2(
            PRETRAIN_ROOT / fold / "pretrained_model.pt",
            STAGING / f"warm_start_{fold}.pt",
        )
    source_layout = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        flat_name = "source__" + relative.replace("/", "__")
        destination = STAGING / flat_name
        shutil.copy2(source, destination)
        source_layout[relative] = flat_name
    inventory = {
        str(path.relative_to(STAGING)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(STAGING.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "run_id": "competition-graph-context-fresh-ensemble-v3",
        "status": "frozen_training_runtime",
        "fresh_split_sha256": EXPECTED["split"],
        "patch_archive_sha256": EXPECTED["archive"],
        "warm_start_sha256": {
            key: EXPECTED[key] for key in ("target_44b6", "target_6bba")
        },
        "planned_model_count": 4,
        "steps_per_model": 20_000,
        "required_visible_gpu_count": 2,
        "maximum_wall_seconds": 34_200,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "metric_hack_allowed": False,
        "submission_command_included": False,
        "source_layout": source_layout,
        "files": inventory,
    }
    (STAGING / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    dataset_metadata = {
        "title": "Biohub Graph Context Fresh Training Runtime v3",
        "id": f"indarkarhana/{RUNTIME_ID}",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n", encoding="ascii"
    )
    return manifest


def build_notebook(runtime_manifest_sha256: str) -> None:
    setup = f'''from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

import torch

RUN_ID = "competition-graph-context-fresh-ensemble-v3"
EXPECTED_RUNTIME_MANIFEST_SHA256 = "{runtime_manifest_sha256}"
EXPECTED_ARCHIVE_SHA256 = "{EXPECTED['archive']}"
EXPECTED_SPLIT_SHA256 = "{EXPECTED['split']}"
WORK = Path("/kaggle/working/graph_context_fresh_v3")
OUTPUT = Path("/kaggle/working/graph_context_fresh_ensemble_v3")

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

runtime_manifests = [
    path
    for path in Path("/kaggle/input").rglob("runtime_manifest.json")
    if sha256_file(path) == EXPECTED_RUNTIME_MANIFEST_SHA256
]
if len(runtime_manifests) != 1:
    raise RuntimeError(
        f"expected one hash-matched runtime manifest, found {{runtime_manifests}}"
    )
RUNTIME = runtime_manifests[0].parent
manifest_path = RUNTIME / "runtime_manifest.json"
manifest = json.loads(manifest_path.read_text())
for relative, record in manifest["files"].items():
    path = RUNTIME / relative
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"runtime file changed: {{relative}}")
if not (
    manifest["patch_archive_sha256"] == EXPECTED_ARCHIVE_SHA256
    and manifest["fresh_split_sha256"] == EXPECTED_SPLIT_SHA256
    and manifest["submission_command_included"] is False
    and manifest["metric_hack_allowed"] is False
    and torch.cuda.is_available()
    and torch.cuda.device_count() == 2
):
    raise RuntimeError("runtime or two-GPU contract failed")
if WORK.exists() or OUTPUT.exists():
    raise RuntimeError("fresh output paths unexpectedly exist")
WORK.mkdir(parents=True)
OUTPUT.mkdir(parents=True)
for relative, flat_name in manifest["source_layout"].items():
    destination = WORK / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNTIME / flat_name, destination)
archive = RUNTIME / "{STAGED_ARCHIVE_NAME}"
with tarfile.open(archive, "r:gz") as stream:
    stream.extractall(WORK, filter="data")
DATA = WORK / "biohub_graph_context_relational_patches_v1"
SPLIT = RUNTIME / "graph_context_fresh_split_v3.json"
TRAINER = WORK / "research/temporal_contrastive/train_graph_context_fresh_ensemble_v3.py"
TERMINAL = RUNTIME / "pretraining_terminal.json"
print(json.dumps({{"gpu_count": torch.cuda.device_count(), "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)], "runtime_verified": True}}, sort_keys=True))
'''
    train = '''SEEDS = (1409101, 1509107)
FOLDS = (("target_44b6", 1), ("target_6bba", 2))
started = time.monotonic()

def launch_worker(seed, fold, index, gpu):
    member = f"seed-{seed}-init-{index}"
    log_path = OUTPUT / f"{member}.log"
    command = [
        sys.executable, str(TRAINER), "worker",
        "--data-root", str(DATA), "--split", str(SPLIT),
        "--initial-model", str(RUNTIME / f"warm_start_{{fold}}.pt"),
        "--warm-start-terminal", str(TERMINAL),
        "--output-root", str(OUTPUT), "--member", member,
        "--seed", str(seed), "--fold", fold,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=WORK, env=environment, stdout=log, stderr=subprocess.STDOUT)
    return member, process, log, log_path

for seed in SEEDS:
    workers = [launch_worker(seed, fold, index, gpu) for gpu, (fold, index) in enumerate(FOLDS)]
    deadline = time.monotonic() + 15_000
    while any(process.poll() is None for _, process, _, _ in workers):
        if time.monotonic() > deadline:
            for _, process, _, _ in workers:
                if process.poll() is None:
                    process.terminate()
            raise TimeoutError(f"worker wave exceeded deadline for seed {seed}")
        time.sleep(20)
    failures = []
    for member, process, log, log_path in workers:
        log.close()
        print(f"{member}: exit={process.returncode}; log={log_path}")
        if process.returncode != 0:
            failures.append(member)
    if failures:
        raise RuntimeError(f"worker failures: {failures}")

aggregate_log = OUTPUT / "aggregate.log"
environment = os.environ.copy()
environment["CUDA_VISIBLE_DEVICES"] = "0"
command = [sys.executable, str(TRAINER), "aggregate", "--data-root", str(DATA), "--split", str(SPLIT), "--output-root", str(OUTPUT)]
with aggregate_log.open("w", encoding="utf-8") as log:
    result = subprocess.run(command, cwd=WORK, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=3_000, check=False)
terminal_path = OUTPUT / "graph_context_fresh_ensemble_terminal.json"
if not terminal_path.is_file():
    raise RuntimeError(f"aggregate produced no terminal; exit={result.returncode}")
terminal = json.loads(terminal_path.read_text())
if result.returncode not in (0, 2) or terminal.get("run_id") != RUN_ID:
    raise RuntimeError(f"aggregate integrity failed; exit={result.returncode}")
terminal["notebook_elapsed_seconds"] = time.monotonic() - started
terminal["required_visible_gpu_count"] = 2
terminal["runtime_manifest_sha256"] = EXPECTED_RUNTIME_MANIFEST_SHA256
terminal["submission_created"] = False
terminal["authorized_for_submission"] = False
terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\\n")
print(json.dumps(terminal, indent=2, sort_keys=True))
'''
    KERNEL_ROOT.mkdir(parents=True, exist_ok=True)
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": False, "language": "python", "sourceType": "notebook", "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            markdown_cell("# Fresh-split graph-context ensemble v3\n\nFour project-authored 74.7M-parameter members train on a hash-frozen competition split. Selection is frozen before audit; no public predictions, leaderboard selection, metric hack, or submission path is present.\n"),
            code_cell(setup),
            code_cell(train),
        ],
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii")
    metadata = {
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Graph Context Fresh Training v3",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "graph-context", "non-replica"],
        "dataset_sources": [f"indarkarhana/{RUNTIME_ID}"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (KERNEL_ROOT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")


def main() -> None:
    build_runtime()
    runtime_hash = sha256_file(STAGING / "runtime_manifest.json")
    build_notebook(runtime_hash)
    print(json.dumps({"runtime": str(STAGING), "runtime_manifest_sha256": runtime_hash, "kernel": str(KERNEL_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
