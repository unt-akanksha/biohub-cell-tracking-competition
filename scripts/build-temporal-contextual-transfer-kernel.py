from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "temporal-contextual-pair-fusion-v3"
KERNEL_ID = "biohub-temporal-contextual-transfer-v3"
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

RUN_ID = "temporal-contextual-pair-fusion-v3"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")
OUTPUT_DIR = Path("/kaggle/working/temporal_contextual_transfer_v3")
DECLARED_BUDGET_SECONDS = 39_600


def write_terminal(status, error=None):
    training_terminal = OUTPUT_DIR / "training_terminal.json"
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": DECLARED_BUDGET_SECONDS,
        "trainer_max_wall_seconds": 36_000,
        "trainer_hard_stop_seconds": 37_800,
        "training_terminal_exists": training_terminal.is_file(),
        "gpu_count_required": 2,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    if error is not None:
        payload["error"] = str(error)
    if training_terminal.is_file():
        payload["training_terminal_sha256"] = hashlib.sha256(
            training_terminal.read_bytes()
        ).hexdigest()
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
print("Contextual transfer watchdog armed for 39,600 seconds.")
'''


SETUP = r'''import importlib
import shutil
import subprocess
import sys
import zipfile

import torch

EXPECTED_RUNTIME_MANIFEST_SHA256 = "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
EXPECTED_ACCEPTANCE_MANIFEST_SHA256 = "cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0"


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        if predicate(payload, path.parent):
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError({"filename": filename, "matches": [str(path) for path in matches]})
    return matches[0]


if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"Contextual transfer requires exactly two GPUs, saw {torch.cuda.device_count()}"
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
synthetic = next(
    (
        path.parent
        for path in input_root.rglob("manifest.json")
        if (path.parent / "sequences").is_dir()
    ),
    None,
)
pretraining_root = unique_json_parent(
    "pretraining_terminal.json",
    lambda payload, parent: (
        payload.get("run_id") == "zebrahub-contextual-pretrain-v1"
        and payload.get("status") == "completed"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and (parent / "target_44b6" / "pretrained_model.pt").is_file()
        and (parent / "target_6bba" / "pretrained_model.pt").is_file()
    ),
)
acceptance_root = unique_json_parent(
    "acceptance_terminal.json",
    lambda payload, _parent: (
        payload.get("run_id") == "zebrahub-contextual-acceptance-evaluation-v1"
        and payload.get("status") == "completed"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and payload.get("acceptance_manifest_sha256")
        == EXPECTED_ACCEPTANCE_MANIFEST_SHA256
        and payload.get("selection_or_checkpoint_redirect_permitted") is False
        and payload.get("competition_data_read") is False
        and payload.get("public_predictions_copied") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ),
)
if None in (runtime_input, competition, support, synthetic):
    raise FileNotFoundError({
        "runtime_input": runtime_input,
        "competition": competition,
        "support": support,
        "synthetic": synthetic,
        "pretraining_root": pretraining_root,
        "acceptance_root": acceptance_root,
    })

runtime = Path("/kaggle/working/temporal_contextual_transfer_runtime_v1")
materialize_runtime_input(runtime_input, runtime)
runtime_manifest_sha256 = hashlib.sha256(
    (runtime / "SOURCE_MANIFEST.json").read_bytes()
).hexdigest()
if runtime_manifest_sha256 != EXPECTED_RUNTIME_MANIFEST_SHA256:
    raise RuntimeError(f"Contextual transfer runtime changed: {runtime_manifest_sha256}")
subprocess.run(
    [
        sys.executable,
        str(runtime / "verify_runtime.py"),
        "--root", str(runtime),
        "--require-gpus",
    ],
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

manifest = json.loads((synthetic / "manifest.json").read_text(encoding="utf-8"))
sequences = [row for row in manifest.get("sequences", []) if int(row.get("T", 0)) >= 4]
if len(sequences) < 2_028:
    raise RuntimeError(f"Synthetic coverage is incomplete: {len(sequences)}")
if any(not (synthetic / str(row["file"])).is_file() for row in sequences):
    raise FileNotFoundError("Synthetic manifest references a missing sequence")
real_geffs = sorted((competition / "train").glob("*.geff"))
real_images = {path.stem for path in (competition / "train").glob("*.zarr")}
eligible_by_prefix = {
    prefix: sum(path.stem.startswith(f"{prefix}_") and path.stem in real_images for path in real_geffs)
    for prefix in ("44b6", "6bba")
}
if eligible_by_prefix != {"44b6": 71, "6bba": 128}:
    raise RuntimeError(f"Real-movie coverage is incomplete: {eligible_by_prefix}")

pretraining_terminal_path = pretraining_root / "pretraining_terminal.json"
pretraining_terminal_sha256 = hashlib.sha256(
    pretraining_terminal_path.read_bytes()
).hexdigest()
acceptance = json.loads(
    (acceptance_root / "acceptance_terminal.json").read_text(encoding="utf-8")
)
if acceptance.get("pretraining_terminal_sha256") != pretraining_terminal_sha256:
    raise RuntimeError("Acceptance and transfer pretraining roots diverge")
for fold in ("target_44b6", "target_6bba"):
    model_path = pretraining_root / fold / "pretrained_model.pt"
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if acceptance["folds"][fold].get("pretrained_model_sha256") != model_sha256:
        raise RuntimeError(f"Acceptance and transfer checkpoint diverge: {fold}")

sys.path.insert(0, str(runtime))
from contextual_pair_fusion import (
    ContextualPairFusionAssociationModel,
    contextual_bidirectional_pair_nll,
)

probe = ContextualPairFusionAssociationModel().to("cuda:0")
probe.load_state_dict(
    torch.load(
        pretraining_root / "target_44b6" / "pretrained_model.pt",
        map_location="cpu",
        weights_only=True,
    ),
    strict=True,
)
probe.train()
probe_patches = torch.randn(176, 3, 17, 17, 17, device="cuda:0")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    embeddings, divisions = probe(probe_patches)
    source_embeddings = embeddings[:48]
    target_embeddings = embeddings[48:]
    source_coords = torch.arange(48, device="cuda:0", dtype=torch.float32)[:, None].repeat(1, 3)
    target_coords = torch.arange(128, device="cuda:0", dtype=torch.float32)[:, None].repeat(1, 3)
    candidates = torch.ones((48, 128), device="cuda:0", dtype=torch.bool)
    positives = torch.zeros_like(candidates)
    positives[torch.arange(48, device="cuda:0"), torch.arange(48, device="cuda:0")] = True
    context = torch.zeros((48, 128, 18), device="cuda:0")
    logits = probe.candidate_pair_logits(
        source_embeddings,
        target_embeddings,
        source_coords,
        target_coords,
        divisions[:48],
        candidates,
        context,
    )
    probe_loss = contextual_bidirectional_pair_nll(logits, positives, candidates)
probe_loss.backward()
if not torch.isfinite(probe_loss):
    raise RuntimeError("Contextual production probe is non-finite")
if sum(parameter.numel() for parameter in probe.parameters()) != 20_747_761:
    raise RuntimeError("Contextual production parameter count changed")
del probe, probe_patches, embeddings, divisions, logits, probe_loss, context
torch.cuda.empty_cache()
print("Contextual 176-patch/6,144-edge pretrained forward-backward passed.")

print(json.dumps({
    "runtime": str(runtime),
    "runtime_manifest_sha256": runtime_manifest_sha256,
    "competition": str(competition),
    "synthetic": str(synthetic),
    "synthetic_sequences": len(sequences),
    "eligible_real_movies": eligible_by_prefix,
    "pretraining_root": str(pretraining_root),
    "pretraining_terminal_sha256": pretraining_terminal_sha256,
    "acceptance_root": str(acceptance_root),
}, indent=2, sort_keys=True))
'''


TRAIN = r'''command = [
    sys.executable,
    str(runtime / "train_dual_fold_contextual_pair_fusion.py"),
    "--orchestrate",
    "--competition-dir", str(competition),
    "--synthetic-root", str(synthetic),
    "--initial-model-root", str(pretraining_root),
    "--output-dir", str(OUTPUT_DIR),
    "--seed", "41027",
    "--steps", "20000",
    "--base-channels", "64",
    "--embedding-channels", "256",
    "--real-replay-probability", "0.60",
    "--learning-rate", "0.00005",
    "--minimum-learning-rate", "0.0000005",
    "--ema-decay", "0.997",
    "--real-train-movies", "96",
    "--real-validation-movies", "12",
    "--real-calibration-movies", "12",
    "--minimum-real-composite-gain", "0.005",
    "--maximum-synthetic-metric-regression", "0.01",
    "--validation-every", "1000",
    "--max-wall-seconds", "36000",
    "--orchestrator-hard-stop-seconds", "37800",
    "--finalization-reserve-seconds", "1200",
]
print("Launching two-GPU contextual transfer:", " ".join(command))
try:
    subprocess.run(command, check=True)
except Exception as error:
    write_terminal("failed", error)
    raise

training_terminal = OUTPUT_DIR / "training_terminal.json"
if not training_terminal.is_file():
    raise RuntimeError("Contextual trainer exited without terminal evidence")
result = json.loads(training_terminal.read_text(encoding="utf-8"))
folds = result.get("folds", {})
valid_folds = bool(
    set(folds) == {"target_44b6", "target_6bba"}
    and all(
        row.get("status") == "completed"
        and row.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
        and int(row.get("best_step", 0)) > 0
        and row.get("finetuning_gate_passed") is True
        and row.get("finetuning_gate", {}).get("passed") is True
        and row.get("initialization", {}).get("policy")
            == "hash-bound ZebraHub external pretraining"
        and row.get("initialization", {}).get("fold") == fold
        and row.get("public_predictions_copied") is False
        and row.get("public_leaderboard_used_for_selection") is False
        and row.get("submission_created") is False
        for fold, row in folds.items()
    )
)
if not (
    result.get("status") == "completed"
    and result.get("run_id") == RUN_ID
    and result.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
    and result.get("gpu_count") == 2
    and result.get("both_folds_trained") is True
    and result.get("both_folds_improved") is True
    and result.get("public_predictions_copied") is False
    and result.get("public_leaderboard_used_for_selection") is False
    and result.get("submission_created") is False
    and valid_folds
):
    raise RuntimeError(f"Invalid contextual transfer terminal: {result}")
subprocess.run(
    [
        sys.executable,
        str(runtime / "verify_appearance_output.py"),
        "--root", str(OUTPUT_DIR),
        "--expected-family", "temporal_contextual_pair_fusion_v3",
        "--strict-checkpoint",
    ],
    check=True,
)
print(json.dumps(result, indent=2, sort_keys=True))
'''


FINISH = r'''FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("Contextual transfer complete; no competition submission was created.")
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
                "# Project-authored contextual transfer v3\n\n"
                "Two independently pretrained contextual models transfer to globally "
                "disjoint reciprocal Biohub folds, one fold per GPU. The frozen ZSNS001 "
                "gate authorizes the family but cannot redirect checkpoints. Fine-tuning "
                "must improve real top-1, MRR, composite, and division recall without "
                "meaningful synthetic regression. No public predictions, leaderboard "
                "selection, artifact creation, or submission is available.\n"
            ),
            code_cell(SETUP),
            code_cell(TRAIN),
            code_cell(FINISH),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{KERNEL_ID}",
        "title": "Biohub Temporal Contextual Transfer v3",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "transfer", "non-replica"],
        "dataset_sources": [
            "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "josefreitasalvesneto/biohub-synthetic-dataset",
            "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
            "indarkarhana/biohub-zebrahub-contextual-acceptance-evaluation-v1",
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
