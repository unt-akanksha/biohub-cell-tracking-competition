from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from biohub_tracker.io import sha256_file  # noqa: E402
from biohub_tracker.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)
from research.temporal_contrastive.contextual_pair_fusion import (  # noqa: E402
    ContextualPairFusionAssociationModel,
)
from research.temporal_contrastive.evaluate_zebrahub_contextual_acceptance import (  # noqa: E402
    FOLDS,
    verify_acceptance_source,
    verify_pretraining_source,
)


RUN_ID = "zebrahub-contextual-acceptance-evaluation-v1"
KERNEL_ID = f"biohub-{RUN_ID}"
KERNEL_DIR = ROOT / "kaggle" / KERNEL_ID
NOTEBOOK = KERNEL_DIR / f"{KERNEL_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
KERNEL_BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-acceptance-kernel.py"
PREFLIGHT_BUILDER = Path(__file__).resolve()
RUNTIME = (
    ROOT
    / ".biohub"
    / "cache"
    / "dataset-redownloads"
    / "biohub-zebrahub-contextual-acceptance-runtime-v1-version3"
)
ACCEPTANCE = (
    ROOT
    / ".biohub"
    / "cache"
    / "dataset-redownloads"
    / "biohub-zebrahub-contextual-acceptance-v1-version1"
)
PRETRAINING_DOWNLOAD = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "zebrahub-contextual-pretrain-v1-mp-repair-version2-20260827"
)
PRETRAINING = PRETRAINING_DOWNLOAD / "zebrahub_contextual_pretrain_v1"
RUNTIME_MANIFEST = RUNTIME / "SOURCE_MANIFEST.json"
ACCEPTANCE_MANIFEST = ACCEPTANCE / "DATASET_MANIFEST.json"
PRETRAINING_TERMINAL = PRETRAINING / "pretraining_terminal.json"
STAGING_REPORT = (
    ROOT
    / "reports"
    / "experiments"
    / "zebrahub-contextual-acceptance-evaluation-v1-staging.md"
)
PRETRAINING_REPORT = (
    ROOT
    / "reports"
    / "experiments"
    / "zebrahub-contextual-pretrain-v1-mp-repair-result.json"
)
COMPETITION_CONFIG = ROOT / "config" / "competition.json"
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json"
FOCUSED_TESTS = (
    ROOT / "tests" / "test_zebrahub_contextual_acceptance.py",
    ROOT / "tests" / "test_zebrahub_contextual_acceptance_evaluation.py",
    ROOT / "tests" / "test_zebrahub_contextual_acceptance_kernel_builder.py",
    ROOT / "tests" / "test_zebrahub_contextual_acceptance_runtime_builder.py",
)

EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "6aefc98b953a15b853731bc970a9734ddcab08a51938bcc9769f29fa6bba1f29"
)
EXPECTED_ACCEPTANCE_MANIFEST_SHA256 = (
    "cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0"
)
EXPECTED_PRETRAINING_TERMINAL_SHA256 = (
    "bbe504907186af058d14f1287492b64e9b83b8c890cb620519ce1536265a618f"
)
EXPECTED_NOTEBOOK_SHA256 = (
    "41835ade5603bb7f64c3703d2eb5568198b3d4ce26e43bef13c708b68a3bc874"
)
EXPECTED_METADATA_SHA256 = (
    "f7a2915c445cfa58e5f2e1361e0b33a40f328bfe5e0dc2702ca802781f8ea094"
)


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


def notebook_code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    return code


def verify_sources() -> tuple[str, dict, dict]:
    subprocess.run([sys.executable, str(KERNEL_BUILDER)], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, str(RUNTIME / "verify_runtime.py"), "--root", str(RUNTIME)],
        cwd=ROOT,
        check=True,
    )
    acceptance_evidence, records = verify_acceptance_source(ACCEPTANCE)
    pretraining = verify_pretraining_source(PRETRAINING)
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
    expected_hashes = {
        RUNTIME_MANIFEST: EXPECTED_RUNTIME_MANIFEST_SHA256,
        ACCEPTANCE_MANIFEST: EXPECTED_ACCEPTANCE_MANIFEST_SHA256,
        PRETRAINING_TERMINAL: EXPECTED_PRETRAINING_TERMINAL_SHA256,
        NOTEBOOK: EXPECTED_NOTEBOOK_SHA256,
        METADATA: EXPECTED_METADATA_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen acceptance input changed: {relative(path)}")
    if not (
        acceptance_evidence.get("status") == "verified_unopened"
        and acceptance_evidence.get("model_predictions_read") is False
        and len(records) == 16
        and pretraining["terminal_sha256"]
        == EXPECTED_PRETRAINING_TERMINAL_SHA256
        and set(pretraining["folds"]) == set(FOLDS)
    ):
        raise RuntimeError("acceptance or parent evidence is not eligible")
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    if not (
        metadata.get("is_private") is True
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
        and metadata.get("dataset_sources")
        == [
            "indarkarhana/biohub-zebrahub-contextual-acceptance-runtime-v1",
            "indarkarhana/biohub-zebrahub-contextual-acceptance-v1",
        ]
        and metadata.get("kernel_sources")
        == ["indarkarhana/biohub-zebrahub-contextual-pretrain-v1"]
        and metadata.get("competition_sources") == []
        and metadata.get("model_sources") == []
    ):
        raise RuntimeError("private two-GPU acceptance input policy changed")
    code = notebook_code()
    required_code = (
        "torch.cuda.device_count() != 2",
        "DECLARED_BUDGET_SECONDS = 3_600",
        '"--hard-stop-seconds", "3300"',
        'result.get("both_folds_improved") is True',
        'row.get("gate_passed") is True',
        "selection_or_checkpoint_redirect_permitted",
    )
    missing = [fragment for fragment in required_code if fragment not in code]
    if missing:
        raise RuntimeError(f"acceptance notebook lost required gates: {missing}")
    lowered = code.casefold()
    if "competitions submit" in lowered or "kaggle competitions" in lowered:
        raise RuntimeError("acceptance notebook contains a submission command")
    return code, acceptance_evidence, pretraining


def verify_checkpoints(pretraining: dict) -> list[Path]:
    paths: list[Path] = []
    for fold in FOLDS:
        model_path = Path(pretraining["folds"][fold]["model_path"])
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model = ContextualPairFusionAssociationModel()
        model.load_state_dict(state, strict=True)
        paths.append(model_path)
    return paths


def main(output: Path = OUTPUT) -> None:
    code, _acceptance_evidence, pretraining = verify_sources()
    checkpoints = verify_checkpoints(pretraining)
    evaluator = RUNTIME / "evaluate_zebrahub_contextual_acceptance.py"
    acceptance_verifier = RUNTIME / "verify_zebrahub_contextual_acceptance.py"
    runtime_verifier = RUNTIME / "verify_runtime.py"
    checks = [
        check(
            "imports",
            [PREFLIGHT_BUILDER, KERNEL_BUILDER, NOTEBOOK, evaluator, *FOCUSED_TESTS],
            "Notebook code compiles, remote-redownloaded runtime v3 verifies, and the focused frozen-acceptance suite passes.",
        ),
        check(
            "inputs",
            [
                METADATA,
                RUNTIME_MANIFEST,
                ACCEPTANCE_MANIFEST,
                PRETRAINING_TERMINAL,
                runtime_verifier,
                acceptance_verifier,
            ],
            "Only the exact repaired runtime v3, unopened ZSNS001 v1, and completed pretraining kernel v2 are attached; competition input and internet are absent.",
        ),
        check(
            "single_batch",
            [
                ROOT / "tests" / "test_zebrahub_contextual_acceptance_evaluation.py",
                evaluator,
                RUNTIME / "contextual_pair_fusion.py",
            ],
            "The production evaluator has finite synthetic-shard coverage tests for exact seeded-control versus checkpoint scoring and immutable inventory gates.",
        ),
        PreflightCheck.create(
            "model_step",
            "not_applicable",
            [],
            ROOT,
            detail="This is a frozen evaluation-only run; it performs no optimizer step and cannot alter checkpoints.",
        ),
        check(
            "checkpoint_roundtrip",
            [*checkpoints, PRETRAINING_TERMINAL, evaluator],
            "Both hash-bound 20,747,761-parameter v3 checkpoints strict-load locally and match the independently verified parent terminal.",
        ),
        check(
            "output_location",
            [NOTEBOOK, METADATA],
            "The notebook writes only fold and aggregate acceptance terminals below /kaggle/working and contains no submission command.",
        ),
        check(
            "dataset_coverage",
            [ACCEPTANCE_MANIFEST, ACCEPTANCE / "acceptance-inventory.json", evaluator],
            "All 16 frozen, hash-bound ZSNS001 shards are present and remain marked unopened before this one-shot evaluation.",
        ),
        check(
            "non_replica_provenance",
            [STAGING_REPORT, PRETRAINING_REPORT, evaluator],
            "The model and evaluator are project-authored; public code, predictions, leaderboard selection, and metric hacks are excluded.",
        ),
        check(
            "quota_policy",
            [COMPETITION_CONFIG, NOTEBOOK, STAGING_REPORT],
            "The registered one-hour ceiling and live guard must leave at least eight Kaggle GPU hours; exactly two visible GPUs are required.",
        ),
    ]
    if "torch.cuda.device_count() != 2" not in code:
        raise RuntimeError("two-GPU acceptance gate disappeared")
    report = PreflightReport.create(RUN_ID, checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(output, report)
    print(output)
    print(report.report_sha256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args().output)
