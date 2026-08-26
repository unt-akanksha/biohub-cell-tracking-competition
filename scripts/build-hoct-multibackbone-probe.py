from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-hoct-probe-finetune.py"
TARGET = ROOT / "kaggle" / "biohub-hoct-multibackbone-probe-v1"
NOTEBOOK = TARGET / "biohub-hoct-multibackbone-probe-v1.ipynb"


def _base() -> dict:
    return runpy.run_path(str(BASE_BUILDER))


def main() -> None:
    source = _base()
    code_cell = source["code_cell"]
    markdown_cell = source["markdown_cell"]

    watchdog = (
        source["WATCHDOG"]
        .replace("hoct-probe-finetune-v1", "hoct-multibackbone-probe-v1")
        .replace("hoct_probe_v1", "hoct_multibackbone_v1")
        .replace("HOCT probe watchdog", "HOCT multibackbone watchdog")
    )
    setup = (
        source["SETUP"]
        .replace("biohub-hoct-runtime-v1", "biohub-hoct-multibackbone-runtime-v1")
        .replace(
            'for name in ("biohub_adapter.py", "train_biohub_hoct_probe.py", "general_v1.pt"):',
            'for name in (\n'
            '    "association_ensemble.py", "biohub_adapter.py", "multibackbone.py",\n'
            '    "train_biohub_hoct_probe.py", "train_biohub_hoct_multibackbone.py",\n'
            '    "general_v1.pt", "ctc_v0.pt",\n'
            '):',
        )
        .replace(
            '"source_model_sha256": hoct_manifest["pretrained_model"]["sha256"],',
            '"source_model_sha256": {name: value["sha256"] for name, value in hoct_manifest["pretrained_models"].items()},',
        )
        .replace(
            'deepcenter = first_existing([\n'
            '    Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),\n'
            '    Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),\n'
            '])',
            'deepcenter = first_existing([\n'
            '    Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),\n'
            '    Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),\n'
            '])\n'
            'trackastra_validation_output = first_existing([\n'
            '    Path("/kaggle/input/biohub-trackastra-raw-confidence-acceptance-v2"),\n'
            '    Path("/kaggle/input/kernels/indarkarhana/biohub-trackastra-raw-confidence-acceptance-v2"),\n'
            '])',
        )
    )
    train = (
        source["TRAIN"]
        .replace("hoct_probe_v1", "hoct_multibackbone_v1")
        .replace("train_biohub_hoct_probe.py", "train_biohub_hoct_multibackbone.py")
        .replace(
            '"--pretrained-model", str(hoct_runtime / "general_v1.pt"),',
            '"--general-model", str(hoct_runtime / "general_v1.pt"),\n'
            '    "--ctc-model", str(hoct_runtime / "ctc_v0.pt"),',
        )
        .replace("Launching Biohub HOCT probe:", "Launching Biohub HOCT multibackbone probe:")
        .replace(
            '"feature_extraction": training["feature_extraction"],',
            '"training_evidence": training["training_evidence"],',
        )
    )
    cached_topology = r'''import math

EXPECTED_PROCESSED_NODES = {
    "44b6_12dfb391": 44139,
    "44b6_267148e4": 21768,
    "6bba_062c8d37": 5812,
    "6bba_07e24132": 26204,
}
EXPECTED_DEEPCENTER_SHA256 = "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"
MAX_REFERENCE_NODE_DRIFT_FRACTION = 0.005


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def processed_node_tolerance(reference_nodes):
    return max(1, math.ceil(reference_nodes * MAX_REFERENCE_NODE_DRIFT_FRACTION))


def validated_cached_topology(root):
    if root is None:
        return None
    launcher_candidates = list(root.rglob("launcher_terminal.json"))
    if not launcher_candidates:
        return None
    valid_launcher = False
    for path in launcher_candidates:
        try:
            terminal = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            terminal.get("run_id") == "trackastra-raw-confidence-acceptance-v2"
            and terminal.get("status") in {"completed", "failed", "aborted", "budget_expired"}
            and terminal.get("submission_created") is False
        ):
            valid_launcher = True
            break
    if not valid_launcher:
        return None
    for report_path in sorted(root.rglob("processed_validation_report.json")):
        csv_path = report_path.with_name("processed_validation.csv")
        if not csv_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            csv_sha256 = sha256_file(csv_path)
        except (OSError, ValueError):
            continue
        expected_hashes = {
            "public_preset_source_sha256": graph_manifest["files"]["public_preset_source.py"]["sha256"],
            "public_config_source_sha256": graph_manifest["files"]["public_config_source.py"]["sha256"],
            "public_postprocess_source_sha256": graph_manifest["files"]["public_postprocess_source.py"]["sha256"],
        }
        checks = [
            report.get("schema_version") == 1,
            report.get("status") == "completed",
            report.get("source_notebook") == "evgendvorkin/biohub-0-927-lb",
            report.get("role") == "frozen comparator postprocessing only",
            report.get("ground_truth_read_for_postprocessing") is False,
            report.get("public_leaderboard_used_for_selection") is False,
            report.get("processed_validation_sha256") == csv_sha256,
            report.get("deepcenter_checkpoint", {}).get("sha256") == EXPECTED_DEEPCENTER_SHA256,
            report.get("deepcenter_checkpoint", {}).get("expected_epoch") == 2,
            all(report.get(key) == value for key, value in expected_hashes.items()),
            all(
                abs(report.get("datasets", {}).get(stem, {}).get("processed_nodes", -count) - count)
                <= processed_node_tolerance(count)
                and report.get("datasets", {}).get(stem, {}).get("expected_processed_nodes") == count
                and report.get("datasets", {}).get(stem, {}).get("processed_node_tolerance")
                == processed_node_tolerance(count)
                for stem, count in EXPECTED_PROCESSED_NODES.items()
            ),
        ]
        if all(checks):
            print(json.dumps({
                "processed_topology_source": "hash_verified_trackastra_v2_kernel_output",
                "processed_validation_csv": str(csv_path),
                "processed_validation_sha256": report["processed_validation_sha256"],
            }, indent=2))
            return csv_path
    return None


cached_processed_csv = validated_cached_topology(trackastra_validation_output)
if cached_processed_csv is None:
    print("No valid cached topology found; using the exact in-kernel materializer.")
'''
    train = train.replace(
        'processed_dir = Path("/kaggle/working/processed_validation")\n'
        'materialize_command = [',
        cached_topology
        + '\nprocessed_dir = Path("/kaggle/working/processed_validation")\n'
        + 'materialize_command = [',
    ).replace(
        'print("Materializing final-topology clean validation:", " ".join(materialize_command))\n'
        'try:\n'
        '    subprocess.run(materialize_command, check=True)\n'
        'except Exception as exc:\n'
        '    write_terminal("failed", exc)\n'
        '    raise',
        'if cached_processed_csv is None:\n'
        '    print("Materializing final-topology clean validation:", " ".join(materialize_command))\n'
        '    try:\n'
        '        subprocess.run(materialize_command, check=True)\n'
        '    except Exception as exc:\n'
        '        write_terminal("failed", exc)\n'
        '        raise\n'
        '    processed_validation_csv = processed_dir / "processed_validation.csv"\n'
        'else:\n'
        '    processed_validation_csv = cached_processed_csv',
    ).replace(
        '"--processed-validation-csv", str(processed_dir / "processed_validation.csv"),',
        '"--processed-validation-csv", str(processed_validation_csv),',
    )
    finish = source["FINISH"].replace(
        "HOCT probe experiment", "HOCT multibackbone experiment"
    )

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
                "# Biohub HOCT multi-backbone association adaptation\n\n"
                "This private experiment fits independent linear probes on the official "
                "general_v1 and CTC-specialized ctc_v0 HOCT backbones. It selects among "
                "four single-model heads and eight ensemble variants using two complete "
                "movies, then reads two disjoint acceptance movies once. It creates no "
                "competition submission and uses no leaderboard feedback. When available, "
                "it reuses the preceding run's hash-verified comparator topology to reserve "
                "the GPU window for HOCT feature extraction.\n"
            ),
            code_cell(setup),
            code_cell(train),
            code_cell(finish),
        ],
    }
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    metadata = {
        "id": "indarkarhana/biohub-hoct-multibackbone-probe-v1",
        "title": "Biohub HOCT Multibackbone Probe v1",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "transformer", "ensemble"],
        "dataset_sources": [
            "indarkarhana/biohub-hoct-multibackbone-runtime-v1",
            "indarkarhana/biohub-trackastra-graph-runtime-v1",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        ],
        "kernel_sources": [
            "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v2"
        ],
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
