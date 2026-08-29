#!/usr/bin/env python
"""Build a Kaggle fallback kernel for external division-policy calibration."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "biohub-external-division-policy-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
RUNTIME_REF = "indarkarhana/biohub-external-division-policy-runtime-v1"
SHARDS_REF = "indarkarhana/biohub-zebrahub-contextual-shards-v1"
PRETRAIN_REF = "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1"
SHARDS_MANIFEST_SHA256 = (
    "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
)


def code_cell(source: str) -> dict:
    ast.parse(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


WATCHDOG = r'''import atexit
import hashlib
import json
import os
import threading
import time
from pathlib import Path

RUN_ID = "external-division-recovery-policy-kaggle-v1"
STARTED = time.monotonic()
FINISHED = False
WATCHDOG_TERMINAL = Path("/kaggle/working/policy-watchdog-terminal.json")

def write_watchdog(status):
    policy = Path("/kaggle/working/policy.json")
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "policy_exists": policy.is_file(),
        "competition_data_read": False,
        "submission_created": False,
    }
    if policy.is_file():
        payload["policy_sha256"] = hashlib.sha256(policy.read_bytes()).hexdigest()
    temporary = WATCHDOG_TERMINAL.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(WATCHDOG_TERMINAL)

def abort_watchdog():
    if not FINISHED:
        write_watchdog("aborted")

atexit.register(abort_watchdog)

def expire_watchdog():
    write_watchdog("budget_expired")
    os._exit(124)

TIMER = threading.Timer(6900, expire_watchdog)
TIMER.daemon = True
TIMER.start()
print("External policy watchdog armed for 6900 seconds")
'''


SETUP_TEMPLATE = r'''import hashlib as _hashlib
import json as _json
import os as _os
import subprocess as _subprocess
import sys as _sys

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
RUNTIME_MANIFEST_SHA256 = "__RUNTIME_MANIFEST_SHA256__"
SHARDS_MANIFEST_SHA256 = "__SHARDS_MANIFEST_SHA256__"

def sha256_file(path):
    digest = _hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

gpu_names = _subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip().splitlines()
if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
    raise RuntimeError(f"Exactly two host T4 GPUs are required, saw {gpu_names}")

runtime_matches = []
for path in INPUT_ROOT.rglob("RUNTIME_MANIFEST.json"):
    if sha256_file(path) != RUNTIME_MANIFEST_SHA256:
        continue
    payload = _json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") == "external-division-policy-runtime-v1":
        runtime_matches.append(path.parent)
if len(runtime_matches) != 1:
    raise RuntimeError(f"Expected one external policy runtime, saw {runtime_matches}")
runtime = runtime_matches[0]
runtime_manifest = _json.loads(
    (runtime / "RUNTIME_MANIFEST.json").read_text(encoding="utf-8")
)
if not (
    runtime_manifest.get("competition_source_allowed") is False
    and runtime_manifest.get("submission_command_included") is False
):
    raise RuntimeError("External policy runtime eligibility changed")
for name, record in runtime_manifest["files"].items():
    if sha256_file(runtime / name) != record["sha256"]:
        raise RuntimeError(f"External policy runtime changed: {name}")

shard_matches = []
for path in INPUT_ROOT.rglob("DATASET_MANIFEST.json"):
    if sha256_file(path) != SHARDS_MANIFEST_SHA256:
        continue
    payload = _json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("run_id") == "zebrahub-contextual-shards-v1"
        and payload.get("competition_test_data_read") is False
        and payload.get("leaderboard_used") is False
    ):
        shard_matches.append(path.parent)
if len(shard_matches) != 1:
    raise RuntimeError(f"Expected one external shard root, saw {shard_matches}")
shards = shard_matches[0]

pretrain_matches = []
for path in INPUT_ROOT.rglob("pretraining_terminal.json"):
    try:
        payload = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if (
        payload.get("status") == "completed"
        and payload.get("run_id") == "zebrahub-multiscale-contextual-pretrain-v1"
        and payload.get("appearance_family")
        == "temporal_multiscale_contextual_pair_fusion_v4"
        and payload.get("gpu_count") == 2
        and payload.get("both_folds_improved") is True
        and payload.get("competition_data_read") is False
        and payload.get("public_leaderboard_used_for_selection") is False
        and payload.get("submission_created") is False
    ):
        pretrain_matches.append(path.parent)
if len(pretrain_matches) != 1:
    raise RuntimeError(f"Expected one verified v4 root, saw {pretrain_matches}")
pretrain = pretrain_matches[0]
print({"gpu_names": gpu_names, "runtime": str(runtime), "shards": str(shards), "pretrain": str(pretrain)})
'''


CALIBRATE = r'''policy_path = WORKING / "policy.json"
environment = dict(_os.environ)
# The frozen calibrator is intentionally one-device. The host allocation still
# has two T4s, while this subprocess exposes one device and loads both folds.
environment["CUDA_VISIBLE_DEVICES"] = "0"
command = [
    _sys.executable,
    str(runtime / "calibrate_division_recovery_policy.py"),
    "--data-root", str(shards),
    "--pretraining-root", str(pretrain),
    "--output", str(policy_path),
    "--patch-batch-size", "32",
]
completed = _subprocess.run(
    command,
    cwd=runtime,
    env=environment,
    check=False,
    capture_output=True,
    text=True,
)
(WORKING / "calibration.stdout.log").write_text(completed.stdout, encoding="utf-8")
(WORKING / "calibration.stderr.log").write_text(completed.stderr, encoding="utf-8")
print(completed.stdout[-12000:])
if completed.returncode != 0:
    raise RuntimeError(
        f"External division policy calibration failed with {completed.returncode}: "
        f"{completed.stderr[-4000:]}"
    )
policy = _json.loads(policy_path.read_text(encoding="utf-8"))
if not (
    policy.get("schema_version") == 1
    and policy.get("status") == "accepted"
    and policy.get("run_id") == "external-division-recovery-policy-v1"
    and policy.get("audit_opened_after_threshold_freeze") is True
    and policy.get("competition_data_read") is False
    and policy.get("public_leaderboard_used_for_selection") is False
    and policy.get("submission_created") is False
    and policy.get("authorized_for_competition_graph_evaluation") is True
    and policy.get("authorized_for_submission") is False
):
    raise RuntimeError("External division policy was not accepted")
policy_terminal = {
    "schema_version": 1,
    "status": "completed",
    "run_id": RUN_ID,
    "host_gpu_count": 2,
    "calibration_visible_gpu_count": 1,
    "policy_sha256": sha256_file(policy_path),
    "frozen_division_logit_threshold": policy["frozen_division_logit_threshold"],
    "selection": policy["selection"],
    "audit": policy["audit"],
    "competition_data_read": False,
    "public_leaderboard_used_for_selection": False,
    "submission_created": False,
    "authorized_for_submission": False,
}
(WORKING / "policy-kernel-terminal.json").write_text(
    _json.dumps(policy_terminal, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(_json.dumps(policy_terminal, indent=2, sort_keys=True))
'''


FINALIZE = r'''FINISHED = True
TIMER.cancel()
write_watchdog("completed")
print("External division policy Kaggle fallback completed without competition data")
'''


def build_notebook(runtime_root: Path) -> dict:
    module = runpy.run_path(
        str(ROOT / "scripts/build-external-division-policy-runtime.py")
    )
    verified = module["verify_runtime"](runtime_root)
    setup = SETUP_TEMPLATE.replace(
        "__RUNTIME_MANIFEST_SHA256__", verified["manifest_sha256"]
    ).replace("__SHARDS_MANIFEST_SHA256__", SHARDS_MANIFEST_SHA256)
    return {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            code_cell(WATCHDOG),
            code_cell(setup),
            code_cell(CALIBRATE),
            code_cell(FINALIZE),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    if TARGET_DIR.exists():
        if not any(TARGET_DIR.iterdir()):
            TARGET_DIR.rmdir()
        else:
            raise FileExistsError(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    TARGET_NOTEBOOK.write_text(
        json.dumps(build_notebook(args.runtime_root), ensure_ascii=True),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub External Division Policy v1",
        "code_file": TARGET_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "external-data", "division-calibration"],
        "dataset_sources": [RUNTIME_REF, SHARDS_REF],
        "kernel_sources": [PRETRAIN_REF],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()
