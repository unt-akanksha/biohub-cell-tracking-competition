from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "lsm-fm-ensemble-validation-v1"
TARGET = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = TARGET / f"biohub-{RUN_ID}.ipynb"
RUNTIME_MANIFEST_SHA256 = "81c70961de251cbd20442cd618fba57b972df7b52de113a2ad9822357b0f3fc1"
FEATURE24_BASE_SHA256 = "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0"
FEATURE36_BASE_SHA256 = "aca3c5d43ef7f3d7ed2ff169d1ab72b71a03acec293a48283d73d38fcf3520e7"
FEATURE24_MODEL_SHA256 = "1a87e6ed6322e0cac98d9c92f7aade2a07bc5dbd4b322c5947f17227f6e256d1"
FEATURE24_RESULT_SHA256 = "da370d810e26253b2a0bed1e16d4bdbc0264ec8a705d7e757a2607fa93c14d0d"
FEATURE24_LAUNCHER_SHA256 = "d1a56bae37975e4221256a79ecbefa7aad2cb1f7f0df7e4e87798f777372ecc6"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).lstrip().splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).lstrip().splitlines(keepends=True),
    }


def exact_sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be an exact SHA-256")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature36-model-sha256", required=True)
    parser.add_argument("--feature36-training-result-sha256", required=True)
    parser.add_argument("--feature36-launcher-sha256", required=True)
    args = parser.parse_args()
    feature36_model_sha256 = exact_sha256(
        args.feature36_model_sha256, "feature36 model"
    )
    feature36_result_sha256 = exact_sha256(
        args.feature36_training_result_sha256, "feature36 training result"
    )
    feature36_launcher_sha256 = exact_sha256(
        args.feature36_launcher_sha256, "feature36 launcher"
    )

    watchdog = r'''
import atexit
import hashlib
import json
import os
import threading
import time
from pathlib import Path

RUN_ID = "lsm-fm-ensemble-validation-v1"
STARTED = time.monotonic()
FINISHED = False
TERMINAL = Path("/kaggle/working/launcher_terminal.json")


def write_terminal(status, error=None):
    result = Path("/kaggle/working/lsm_fm_ensemble/lsm_fm_ensemble_validation.json")
    payload = {
        "run_id": RUN_ID,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - STARTED, 3),
        "declared_budget_seconds": 3600,
        "hard_stop_seconds": 3300,
        "result_exists": result.is_file(),
        "public_predictions_copied": False,
        "competition_submission_performed": False,
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
TIMER = threading.Timer(3300, budget_expired)
TIMER.daemon = True
TIMER.start()
print("LSM-FM ensemble watchdog armed for 3,300 seconds.")
'''

    setup = f'''
import importlib
import subprocess
import sys

RUNTIME_MANIFEST_SHA256 = "{RUNTIME_MANIFEST_SHA256}"
FEATURE24_BASE_SHA256 = "{FEATURE24_BASE_SHA256}"
FEATURE36_BASE_SHA256 = "{FEATURE36_BASE_SHA256}"
FEATURE24_MODEL_SHA256 = "{FEATURE24_MODEL_SHA256}"
FEATURE24_RESULT_SHA256 = "{FEATURE24_RESULT_SHA256}"
FEATURE24_LAUNCHER_SHA256 = "{FEATURE24_LAUNCHER_SHA256}"
FEATURE36_MODEL_SHA256 = "{feature36_model_sha256}"
FEATURE36_RESULT_SHA256 = "{feature36_result_sha256}"
FEATURE36_LAUNCHER_SHA256 = "{feature36_launcher_sha256}"


def first_existing(candidates):
    return next((path for path in candidates if path.exists()), None)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


input_root = Path("/kaggle/input")
runtime = first_existing([
    Path("/kaggle/input/datasets/indarkarhana/biohub-lsm-fm-ensemble-runtime-v1"),
    Path("/kaggle/input/biohub-lsm-fm-ensemble-runtime-v1"),
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
feature24_output = first_existing([
    Path("/kaggle/input/notebooks/indarkarhana/biohub-lsm-fm-pu-adaptation-v2"),
    Path("/kaggle/input/kernels/indarkarhana/biohub-lsm-fm-pu-adaptation-v2"),
    Path("/kaggle/input/biohub-lsm-fm-pu-adaptation-v2"),
])
feature36_output = first_existing([
    Path("/kaggle/input/notebooks/indarkarhana/biohub-lsm-fm-image-text-pu-adaptation-v1"),
    Path("/kaggle/input/kernels/indarkarhana/biohub-lsm-fm-image-text-pu-adaptation-v1"),
    Path("/kaggle/input/biohub-lsm-fm-image-text-pu-adaptation-v1"),
])
if None in (runtime, graph_runtime, competition, support, feature24_output, feature36_output):
    raise FileNotFoundError({{
        "runtime": runtime,
        "graph_runtime": graph_runtime,
        "competition": competition,
        "support": support,
        "feature24_output": feature24_output,
        "feature36_output": feature36_output,
    }})

numpy_before = importlib.import_module("numpy").__version__
if numpy_before != "2.0.2":
    raise RuntimeError(f"unexpected Kaggle NumPy before offline install: {{numpy_before}}")
wheel_dirs = sorted({{path.parent for path in support.rglob("*.whl")}})
if not wheel_dirs:
    raise FileNotFoundError("support pack contains no dependency wheels")
pip_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
for wheel_dir in wheel_dirs:
    pip_cmd.extend(["--find-links", str(wheel_dir)])
pip_cmd.extend([
    "bidict==0.23.1", "donfig==0.8.1.post1", "geff==1.2.0.1.1",
    "geff-spec==1.1.1", "ilpy==0.6.0", "numcodecs==0.15.1",
    "polars==1.42.0", "polars-runtime-32==1.42.0", "pyscipopt==6.2.1",
    "rustworkx==0.18.0", "tracksdata==0.1.0rc6.dev3+g980c2d30a", "zarr==3.2.1",
])
subprocess.run(pip_cmd, check=True)

support_repo = next((path for path in support.rglob("biohub_tracking") if path.is_dir()), None)
if support_repo is None:
    raise FileNotFoundError("biohub_tracking source not found in support pack")
manifest_path = runtime / "SOURCE_MANIFEST.json"
if sha256_file(manifest_path) != RUNTIME_MANIFEST_SHA256:
    raise RuntimeError("ensemble runtime manifest hash mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
monai_archive = runtime / "monai-1.5.1.zip"
monai_mount = runtime / manifest["archives"]["monai-1.5.1.zip"]["mount_directory"]
if monai_archive.is_file():
    if sha256_file(monai_archive) != manifest["files"]["monai-1.5.1.zip"]["sha256"]:
        raise RuntimeError("MONAI archive hash mismatch")
    monai_import_root = monai_archive
elif monai_mount.is_dir():
    for name, row in manifest["archives"]["monai-1.5.1.zip"]["files"].items():
        path = monai_mount / name
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"extracted MONAI file hash mismatch: {{name}}")
    monai_import_root = monai_mount
else:
    raise FileNotFoundError("MONAI archive or expanded resource is missing")
sys.path.insert(0, str(monai_import_root))
sys.path.insert(0, str(runtime))
sys.path.insert(0, str(support_repo.parent))
for module in ("numpy", "scipy", "torch", "monai", "tracksdata", "zarr"):
    imported = importlib.import_module(module)
    print(module, getattr(imported, "__version__", "unknown"))
if importlib.import_module("numpy").__version__ != numpy_before:
    raise RuntimeError("offline dependency installation changed NumPy")
torch = importlib.import_module("torch")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

for name, row in manifest["files"].items():
    path = runtime / name
    if name == "monai-1.5.1.zip" and monai_mount.is_dir():
        continue
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"runtime file hash mismatch: {{name}}")
feature24_base = runtime / "lsm_fm_image_only_student.pt"
feature36_base = runtime / "lsm_fm_image_text_student.pt"
if sha256_file(feature24_base) != FEATURE24_BASE_SHA256:
    raise RuntimeError("feature-24 base checkpoint mismatch")
if sha256_file(feature36_base) != FEATURE36_BASE_SHA256:
    raise RuntimeError("feature-36 base checkpoint mismatch")


def locate_source_evidence(root, run_id, launcher_sha, model_sha, result_sha):
    launchers = [path for path in root.rglob("launcher_terminal.json") if sha256_file(path) == launcher_sha]
    models = [path for path in root.rglob("best.pt") if sha256_file(path) == model_sha]
    results = [path for path in root.rglob("pu_training_result.json") if sha256_file(path) == result_sha]
    if len(launchers) != 1 or len(models) != 1 or len(results) != 1:
        raise FileNotFoundError({{
            "run_id": run_id,
            "launchers": [str(path) for path in launchers],
            "models": [str(path) for path in models],
            "results": [str(path) for path in results],
        }})
    launcher = json.loads(launchers[0].read_text(encoding="utf-8"))
    if not (
        launcher.get("run_id") == run_id
        and launcher.get("status") == "completed"
        and launcher.get("competition_submission_performed") is False
        and launcher.get("public_predictions_copied") is False
        and launcher.get("training_result_sha256") == result_sha
    ):
        raise ValueError(f"invalid source launcher evidence for {{run_id}}")
    result = json.loads(results[0].read_text(encoding="utf-8"))
    if not (
        result.get("status") == "completed"
        and result.get("validation_overlap") == []
        and result.get("public_predictions_copied") is False
        and result.get("best_weight_sha256") == model_sha
    ):
        raise ValueError(f"invalid source training evidence for {{run_id}}")
    return models[0], results[0]


feature24_model, feature24_result = locate_source_evidence(
    feature24_output,
    "lsm-fm-pu-adaptation-v2",
    FEATURE24_LAUNCHER_SHA256,
    FEATURE24_MODEL_SHA256,
    FEATURE24_RESULT_SHA256,
)
feature36_model, feature36_result = locate_source_evidence(
    feature36_output,
    "lsm-fm-image-text-pu-adaptation-v1",
    FEATURE36_LAUNCHER_SHA256,
    FEATURE36_MODEL_SHA256,
    FEATURE36_RESULT_SHA256,
)
probe = importlib.import_module("tracksdata").graph.IndexedRXGraph.from_geff(
    graph_runtime / "validator_raw" / "44b6_12dfb391.geff"
)
probe = probe[0] if isinstance(probe, tuple) else probe
if probe.node_attrs().height <= 0 or probe.edge_attrs().height <= 0:
    raise RuntimeError("baseline graph IO probe is empty")
print(json.dumps({{
    "runtime": str(runtime),
    "competition": str(competition),
    "feature24_model": str(feature24_model),
    "feature36_model": str(feature36_model),
    "cuda_devices_visible": torch.cuda.device_count(),
    "cuda_device_0": torch.cuda.get_device_name(0),
}}, indent=2))
'''

    run = r'''
output_dir = Path("/kaggle/working/lsm_fm_ensemble")
command = [
    sys.executable,
    str(runtime / "evaluate_lsm_fm_ensemble.py"),
    "--competition-dir", str(competition),
    "--feature24-base", str(feature24_base),
    "--feature24-base-sha256", FEATURE24_BASE_SHA256,
    "--feature24-model", str(feature24_model),
    "--feature24-training-result", str(feature24_result),
    "--feature36-base", str(feature36_base),
    "--feature36-base-sha256", FEATURE36_BASE_SHA256,
    "--feature36-model", str(feature36_model),
    "--feature36-training-result", str(feature36_result),
    "--baseline-predictions", str(graph_runtime / "validator_raw"),
    "--output-dir", str(output_dir),
    "--batch-size", "1",
    "--calibration-frames", "12",
    "--max-wall-seconds", "3000",
]
run_env = os.environ.copy()
run_env["PYTHONPATH"] = os.pathsep.join([
    str(runtime), str(monai_import_root), str(support_repo.parent),
    run_env.get("PYTHONPATH", ""),
]).rstrip(os.pathsep)
print("Launching clean LSM-FM ensemble validation:", " ".join(command))
try:
    subprocess.run(command, check=True, env=run_env)
except Exception as exc:
    write_terminal("validation_failed", exc)
    raise
result_path = output_dir / "lsm_fm_ensemble_validation.json"
if not result_path.is_file():
    raise RuntimeError("ensemble evaluator exited without terminal evidence")
result = json.loads(result_path.read_text(encoding="utf-8"))
print(json.dumps({
    "selected_candidate": result["selected_candidate"],
    "selection_passed": result["selection_passed"],
    "acceptance_opened": result["acceptance_opened"],
    "promotion_passed": result["promotion_passed"],
}, indent=2, sort_keys=True))
'''

    finish = r'''
unexpected_submissions = [
    path for path in Path("/kaggle/working").rglob("*")
    if path.is_file() and path.name.lower() in {"submission.csv", "submission.zip"}
]
if unexpected_submissions:
    raise RuntimeError(f"Validation unexpectedly created submission artifacts: {unexpected_submissions}")
FINISHED = True
TIMER.cancel()
write_terminal("completed")
print("LSM-FM ensemble clean validation complete; no submission was created.")
'''

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
            code_cell(watchdog),
            markdown_cell(
                """
                # Clean feature-diverse LSM-FM ensemble validation

                This private experiment compares independently adapted official
                LSM-FM feature-24 and feature-36 students with one fixed 0.5/0.5
                probability ensemble. One global candidate is selected on eight
                movies before four acceptance movies can be read. It uses no
                leaderboard feedback, copies no public predictions or Kaggle
                code, and cannot create a competition submission.
                """
            ),
            code_cell(setup),
            code_cell(run),
            code_cell(finish),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": f"indarkarhana/biohub-{RUN_ID}",
        "title": "Biohub LSM-FM Ensemble Validation v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "swinunetr", "ensemble", "clean-validation"],
        "dataset_sources": [
            "indarkarhana/biohub-lsm-fm-ensemble-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-lsm-fm-pu-adaptation-v2",
            "indarkarhana/biohub-lsm-fm-image-text-pu-adaptation-v1",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaTeslaT4",
    }
    (TARGET / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
