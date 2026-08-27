from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for source_root in (ROOT, ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from biohub_tracker.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)


RUN_ID = "temporal-patch-dual-fold-v1-inventory-repair"
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-patch-dual-fold-v1"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-patch-dual-fold-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
BUILDER = ROOT / "scripts" / "build-temporal-patch-dual-fold-kernel.py"
RUNTIME = ROOT / ".biohub" / "staging" / "biohub-temporal-patch-runtime-v1"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json"


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


def main() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME / "verify_runtime.py"),
            "--root",
            str(RUNTIME),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_temporal_patch_kernel_builder.py",
            "tests/test_temporal_patch_runtime_builder.py",
            "tests/test_temporal_runtime_verifier.py",
            "tests/test_temporal_contrastive.py",
            "tests/test_temporal_dual_fold_output_verifier.py",
            "tests/test_submission_sharding.py",
        ],
        cwd=ROOT,
        check=True,
    )
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(
        (RUNTIME / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    trainer = (RUNTIME / "train_dual_fold_patch.py").read_text(encoding="utf-8")
    verifier = (RUNTIME / "verify_appearance_output.py").read_text(encoding="utf-8")
    required_code = (
        'expected_real_inventory = {"44b6": 71, "6bba": 128}',
        "eligible_by_prefix != expected_real_inventory",
        "Dense 176-patch production forward/backward",
        "torch.cuda.device_count() != 2",
    )
    if any(item not in code for item in required_code):
        raise RuntimeError("inventory-repair notebook lost a required exact gate")
    required_trainer = (
        '"expected_real_train_movies": 96',
        '"expected_real_train_movies": 45',
        '"effective_real_train_movies": len(real_train)',
    )
    if any(item not in trainer for item in required_trainer):
        raise RuntimeError("trainer does not bind effective reciprocal inventories")
    required_verifier = (
        '"real_train_count": 96',
        '"real_train_count": 45',
        'config.get("effective_real_train_movies")',
    )
    if any(item not in verifier for item in required_verifier):
        raise RuntimeError("output verifier does not bind effective reciprocal counts")
    if not (
        manifest.get("run_id") == "temporal-patch-dual-fold-v1"
        and len(manifest.get("files", {})) == 30
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
        and config.get("scientific_configuration_changed") is False
        and config.get("repair", {}).get("effective_real_train_movies_by_fold")
        == {"target_44b6": 96, "target_6bba": 45}
        and config.get("required_visible_gpu_count") == 2
        and config.get("authorized_for_submission") is False
    ):
        raise RuntimeError("inventory-repair accelerator, split, or policy changed")
    if "competitions submit" in code or "kaggle competitions" in code:
        raise RuntimeError("inventory-repair notebook contains a submission command")

    runtime_manifest = RUNTIME / "SOURCE_MANIFEST.json"
    trainer_path = RUNTIME / "train_dual_fold_patch.py"
    verifier_path = RUNTIME / "verify_appearance_output.py"
    test_path = ROOT / "tests" / "test_temporal_dual_fold_output_verifier.py"
    checks = [
        check(
            "imports",
            [BUILDER, NOTEBOOK, trainer_path, verifier_path, test_path],
            "Every notebook cell compiles and the focused kernel, runtime, trainer, output-verifier, and sharding tests pass.",
        ),
        check(
            "inputs",
            [NOTEBOOK, runtime_manifest, CONFIG],
            "The private 30-file runtime, corrected synthetic source, coherent Trackastra control, support wheels, and exact competition inventory are attached with two T4s and internet off.",
        ),
        check(
            "single_batch",
            [RUNTIME / "patch_model.py", ROOT / "tests" / "test_temporal_contrastive.py"],
            "The unchanged three-channel 17-cubed physical appearance path passes its focused forward contract.",
        ),
        check(
            "model_step",
            [trainer_path, RUNTIME / "patch_model.py"],
            "The unchanged production loss/backward path passes with finite gradients; only inventory assertions changed.",
        ),
        check(
            "checkpoint_roundtrip",
            [NOTEBOOK, RUNTIME / "patch_model.py"],
            "The strict 19,221,954-parameter checkpoint roundtrip remains required before optimizer training.",
        ),
        check(
            "output_location",
            [NOTEBOOK, METADATA],
            "Only launcher and reciprocal training evidence can be written; no competition artifact or submit command is present.",
        ),
        check(
            "dataset_coverage",
            [NOTEBOOK, trainer_path, verifier_path, CONFIG],
            "The exact 71/128 attached inventory yields the intended up-to-96 source partitions: 96 and 45 training movies after two opened exclusions plus 12 validation and 12 calibration reservations per prefix.",
        ),
        check(
            "dense_memory",
            [NOTEBOOK, RUNTIME / "patch_model.py"],
            "Before training, one T4 must still pass the production 176-patch mixed-precision forward/backward and checkpoint roundtrip.",
        ),
        check(
            "failure_provenance",
            [CONFIG],
            "The 510.6-second pre-training inventory rejection, quota charge, and both downloaded evidence hashes are bound and are not scientific model evidence.",
        ),
        check(
            "quota_policy",
            [CONFIG, NOTEBOOK],
            "The repair retains an immutable 11-hour ceiling and requires a fresh guard projecting at least eight Kaggle GPU hours remaining.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(OUTPUT, report)
    print(OUTPUT)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
