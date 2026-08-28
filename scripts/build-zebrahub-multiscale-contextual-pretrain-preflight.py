#!/usr/bin/env python
"""Preflight the post-v3 46.4M-parameter ZebraHub multiscale lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from biohub_tracker.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)
from research.temporal_contrastive.contextual_pair_fusion import (  # noqa: E402
    ContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.multiscale_contextual_pair_fusion import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    MultiscaleContextualPairFusionAssociationModel,
)


RUN_ID = "zebrahub-multiscale-contextual-pretrain-v1"
KERNEL_ID = "biohub-zebrahub-multiscale-pretrain-v1"
KERNEL_DIR = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = KERNEL_DIR / f"{KERNEL_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
KERNEL_BUILDER = (
    ROOT / "scripts" / "build-zebrahub-multiscale-contextual-pretrain-kernel.py"
)
PREFLIGHT_BUILDER = Path(__file__).resolve()
RUNTIME = (
    ROOT
    / ".biohub"
    / "cache"
    / "dataset-redownloads"
    / "biohub-temporal-multiscale-contextual-runtime-v4-version3"
)
DATASET = (
    ROOT
    / ".biohub"
    / "cache"
    / "dataset-redownloads"
    / "biohub-zebrahub-contextual-shards-v1-version4"
)
V3_DOWNLOAD = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "zebrahub-contextual-pretrain-v1-mp-repair-version2-20260827"
)
V3_ROOT = V3_DOWNLOAD / "zebrahub_contextual_pretrain_v1"
V3_LAUNCHER = V3_DOWNLOAD / "launcher_terminal.json"
V3_TERMINAL = V3_ROOT / "pretraining_terminal.json"
ACCEPTANCE_DOWNLOAD = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "zsns001-contextual-gate-v1-version1-20260828"
)
ACCEPTANCE_ROOT = ACCEPTANCE_DOWNLOAD / "zebrahub_contextual_acceptance_v1"
ACCEPTANCE_LAUNCHER = ACCEPTANCE_DOWNLOAD / "launcher_terminal.json"
ACCEPTANCE_TERMINAL = ACCEPTANCE_ROOT / "acceptance_terminal.json"
OUTPUT = (
    ROOT
    / "artifacts"
    / "preflights"
    / f"{RUN_ID}-post-v3-autolaunch-v1.json"
)
CONFIG = (
    ROOT
    / "config"
    / "experiments"
    / "temporal-multiscale-contextual-pair-fusion-v4.json"
)
TECHNIQUE_REPORT = (
    ROOT / "reports" / "experiments" / "biohub-technique-scan-2026-08-27.md"
)

EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
)
EXPECTED_NOTEBOOK_SHA256 = (
    "c15f7e1d9080067bec78e7567b1aba8c95132b135854cedd56951954083ed4ec"
)
EXPECTED_METADATA_SHA256 = (
    "d7edfb27e13d11c7cc3d9e85d3683417193e6acafc4ed3c0eb13362ef39504c6"
)
EXPECTED_V3_TERMINAL_SHA256 = (
    "bbe504907186af058d14f1287492b64e9b83b8c890cb620519ce1536265a618f"
)
EXPECTED_V3_LAUNCHER_SHA256 = (
    "173639354f113029b71eb8d2c86827915f50d1e057aa3ea7ba1a504921e42834"
)
EXPECTED_ACCEPTANCE_SHA256 = (
    "2c886d45849330c30c5dc084aafeb1bff0436ce80794e56420b92e7273bf5da9"
)
EXPECTED_ACCEPTANCE_LAUNCHER_SHA256 = (
    "91c3217b8848339dca208dbb41deecbb81f52260269150acca93a24bf551a4f3"
)
FOLDS = {"target_44b6", "target_6bba"}
FOCUSED_TESTS = (
    ROOT / "tests" / "test_multiscale_contextual_pair_fusion.py",
    ROOT / "tests" / "test_zebrahub_multiscale_contextual_pretrain.py",
    ROOT / "tests" / "test_multiscale_contextual_trainer.py",
    ROOT / "tests" / "test_temporal_patch_runtime_builder.py",
    ROOT / "tests" / "test_zebrahub_multiscale_contextual_pretrain_kernel_builder.py",
    ROOT / "tests" / "test_temporal_multiscale_contextual_transfer_kernel_builder.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def check(name: str, evidence: list[Path], detail: str) -> PreflightCheck:
    return PreflightCheck.create(
        name,
        "passed",
        [relative(path) for path in evidence],
        ROOT,
        detail=detail,
    )


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} changed: {observed}")


def verify_notebook() -> str:
    subprocess.run([sys.executable, str(KERNEL_BUILDER)], cwd=ROOT, check=True)
    require_hash(NOTEBOOK, EXPECTED_NOTEBOOK_SHA256, "multiscale notebook")
    require_hash(METADATA, EXPECTED_METADATA_SHA256, "multiscale metadata")
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    if not (
        metadata.get("id") == "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1"
        and metadata.get("is_private") is True
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
        and metadata.get("dataset_sources")
        == [
            "indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4",
            "indarkarhana/biohub-zebrahub-contextual-shards-v1",
        ]
        and metadata.get("kernel_sources")
        == [
            "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
            "indarkarhana/biohub-zsns001-contextual-gate-v1",
        ]
        and metadata.get("competition_sources") == []
    ):
        raise RuntimeError("multiscale pretraining input policy changed")
    required = (
        "torch.cuda.device_count() != 2",
        "DECLARED_BUDGET_SECONDS = 24_000",
        '"--orchestrator-hard-stop-seconds", "22800"',
        '"--max-wall-seconds", "21600"',
        "Both accepted v3 checkpoints strict-loaded",
        "recompute_acceptance_gate",
        "selection_gate_passed",
        "audit_gate_passed",
        'result.get("both_folds_improved") is True',
    )
    missing = [fragment for fragment in required if fragment not in code]
    if missing:
        raise RuntimeError(f"multiscale notebook lost required gates: {missing}")
    if "competitions submit" in code.casefold() or "kaggle competitions" in code.casefold():
        raise RuntimeError("multiscale notebook contains a submission command")
    return code


def verify_runtime_and_dataset() -> dict[str, Any]:
    require_hash(
        RUNTIME / "SOURCE_MANIFEST.json",
        EXPECTED_RUNTIME_MANIFEST_SHA256,
        "multiscale runtime manifest",
    )
    require_hash(
        DATASET / "DATASET_MANIFEST.json",
        EXPECTED_DATASET_MANIFEST_SHA256,
        "ZebraHub dataset manifest",
    )
    subprocess.run(
        [sys.executable, str(RUNTIME / "verify_runtime.py"), "--root", str(RUNTIME)],
        cwd=ROOT,
        check=True,
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(RUNTIME / "verify_zebrahub_contextual_dataset.py"),
            "--root",
            str(DATASET),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(verified.stdout)
    if not (
        payload.get("status") == "verified"
        and payload.get("manifest_sha256") == EXPECTED_DATASET_MANIFEST_SHA256
        and payload.get("training", {}).get("shards") == 64
        and payload.get("validation", {}).get("shards") == 16
        and payload.get("competition_test_data_read") is False
        and payload.get("leaderboard_used") is False
        and payload.get("submission_created") is False
    ):
        raise RuntimeError("ZebraHub dataset verification changed")
    return payload


def verify_accepted_v3() -> None:
    require_hash(V3_TERMINAL, EXPECTED_V3_TERMINAL_SHA256, "v3 pretraining terminal")
    require_hash(V3_LAUNCHER, EXPECTED_V3_LAUNCHER_SHA256, "v3 launcher")
    require_hash(ACCEPTANCE_TERMINAL, EXPECTED_ACCEPTANCE_SHA256, "ZSNS001 acceptance")
    require_hash(
        ACCEPTANCE_LAUNCHER,
        EXPECTED_ACCEPTANCE_LAUNCHER_SHA256,
        "ZSNS001 launcher",
    )
    v3 = json.loads(V3_TERMINAL.read_text(encoding="utf-8"))
    accepted = json.loads(ACCEPTANCE_TERMINAL.read_text(encoding="utf-8"))
    if not (
        v3.get("status") == "completed"
        and v3.get("run_id") == "zebrahub-contextual-pretrain-v1"
        and v3.get("appearance_family") == "temporal_contextual_pair_fusion_v3"
        and v3.get("gpu_count") == 2
        and v3.get("both_folds_improved") is True
        and set(v3.get("folds", {})) == FOLDS
        and accepted.get("status") == "completed"
        and accepted.get("run_id") == "zebrahub-contextual-acceptance-evaluation-v1"
        and accepted.get("acceptance_source") == "ZSNS001"
        and accepted.get("gpu_count") == 2
        and accepted.get("both_folds_improved") is True
        and accepted.get("pretraining_terminal_sha256") == EXPECTED_V3_TERMINAL_SHA256
        and set(accepted.get("folds", {})) == FOLDS
        and v3.get("competition_data_read") is False
        and accepted.get("competition_data_read") is False
        and v3.get("public_leaderboard_used_for_selection") is False
        and accepted.get("public_leaderboard_used_for_selection") is False
        and v3.get("submission_created") is False
        and accepted.get("submission_created") is False
    ):
        raise RuntimeError("accepted contextual-v3 warm-start chain is invalid")
    for fold in FOLDS:
        model_path = V3_ROOT / fold / "pretrained_model.pt"
        expected_model = str(v3["folds"][fold]["model_sha256"])
        require_hash(model_path, expected_model, f"v3 checkpoint {fold}")
        row = accepted["folds"][fold]
        if not (
            row.get("gate_passed") is True
            and row.get("pretrained_model_sha256") == expected_model
            and row.get("pretraining_terminal_sha256") == EXPECTED_V3_TERMINAL_SHA256
        ):
            raise RuntimeError(f"ZSNS001 fold evidence changed: {fold}")
        model = ContextualPairFusionAssociationModel()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True), strict=True
        )
        if sum(parameter.numel() for parameter in model.parameters()) != 20_747_761:
            raise RuntimeError(f"v3 architecture changed: {fold}")
        del model


def main(output: Path = OUTPUT) -> None:
    verify_notebook()
    dataset = verify_runtime_and_dataset()
    verify_accepted_v3()
    multiscale = MultiscaleContextualPairFusionAssociationModel()
    if sum(parameter.numel() for parameter in multiscale.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("multiscale parameter inventory changed")
    del multiscale
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *[relative(path) for path in FOCUSED_TESTS],
        ],
        cwd=ROOT,
        check=True,
    )
    checks = [
        check(
            "imports",
            [PREFLIGHT_BUILDER, KERNEL_BUILDER, NOTEBOOK, *FOCUSED_TESTS],
            "The deterministic notebook compiles and the multiscale architecture, trainer, runtime, and kernel suites pass.",
        ),
        check(
            "inputs",
            [METADATA, RUNTIME / "SOURCE_MANIFEST.json", DATASET / "DATASET_MANIFEST.json"],
            "Only verified private runtime v3, ZebraHub shards v4, accepted contextual-v3 weights, and ZSNS001 acceptance are attached; competition data is absent.",
        ),
        check(
            "model_step",
            [RUNTIME / "multiscale_contextual_pair_fusion.py", RUNTIME / "train_zebrahub_multiscale_contextual_pretrain.py"],
            "The 46,386,607-parameter project-authored residual model has a finite forward/backward test and starts from strict accepted-v3 weights.",
        ),
        check(
            "checkpoint_roundtrip",
            [V3_TERMINAL, ACCEPTANCE_TERMINAL, *[V3_ROOT / fold / "pretrained_model.pt" for fold in sorted(FOLDS)]],
            "Both 20,747,761-parameter parent checkpoints strict-load and match the passing untouched-ZSNS001 acceptance hashes.",
        ),
        check(
            "dataset_coverage",
            [DATASET / "train-inventory.json", DATASET / "validation-inventory.json"],
            f"All {dataset['training']['shards']} ZSNS004 training and {dataset['validation']['shards']} disjoint ZSNS005 validation shards verify without competition data.",
        ),
        check(
            "non_replica_provenance",
            [CONFIG, TECHNIQUE_REPORT, RUNTIME / "multiscale_contextual_pair_fusion.py"],
            "The axial multiscale branch is project-authored; public code/predictions, metric hacks, and leaderboard selection remain excluded.",
        ),
        check(
            "quota_policy",
            [NOTEBOOK, CONFIG],
            "The stage requires exactly two T4 GPUs and a 24,000-second ceiling; post-v3 orchestration may use all then-available quota with zero reserve.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(output, report)
    print(output)
    print(report.report_sha256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.output)
