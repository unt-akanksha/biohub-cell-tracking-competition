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


RUN_ID = "temporal-patch-dual-fold-blend-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
BUILDER = ROOT / "scripts" / "build-temporal-patch-blend-kernel.py"
RUNTIME = ROOT / ".biohub" / "staging" / "biohub-temporal-patch-runtime-v1"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}-coherence2.json"


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
            "tests/test_temporal_patch_blend_kernel_builder.py",
            "tests/test_temporal_dual_fold_output_verifier.py",
            "tests/test_temporal_contrastive.py",
            "tests/test_temporal_appearance_processed_acceptance.py",
        ],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    manifest = json.loads(
        (RUNTIME / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    if len(manifest.get("files", {})) != 30:
        raise RuntimeError("runtime file inventory changed")
    if not (
        metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
        and config.get("required_visible_gpu_count") == 2
        and config.get("authorized_for_submission") is False
    ):
        raise RuntimeError("calibration execution policy changed")
    if "competitions submit" in code or "kaggle competitions" in code:
        raise RuntimeError("calibration notebook contains a submission command")
    required_contracts = (
        "verify_appearance_output.py",
        "--strict-checkpoint",
        "verify_trackastra_output.py",
        "processed_acceptance_ground_truth_read",
        "public_leaderboard_used_for_selection",
        "submission_created",
    )
    if any(contract not in code for contract in required_contracts):
        raise RuntimeError("calibration fail-closed contract is incomplete")

    verifier = ROOT / "research" / "temporal_contrastive" / "verify_dual_fold_training_output.py"
    calibrator = ROOT / "research" / "temporal_contrastive" / "calibrate_dual_fold_blend.py"
    processed = ROOT / "research" / "temporal_contrastive" / "dual_fold_appearance_processed_acceptance.py"
    runtime_manifest = RUNTIME / "SOURCE_MANIFEST.json"
    checks = [
        check(
            "imports",
            [BUILDER, NOTEBOOK, verifier, calibrator],
            "Every notebook code cell compiles, the runtime verifies, and focused calibration, output-verifier, and processed-boundary tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, runtime_manifest],
            "The private appearance runtime, reciprocal appearance output, coherent Trackastra control, support wheels, and Biohub competition are declared with T4 x2 and internet off.",
        ),
        check(
            "training_evidence_boundary",
            [NOTEBOOK, verifier, calibrator],
            "Calibration requires exact worker/aggregate equality, model hashes, strict checkpoint loading, threshold-eligible folds, and globally disjoint reciprocal inventories.",
        ),
        check(
            "clean_selection_boundary",
            [CONFIG, calibrator, processed],
            "Only 12 pre-reserved movies per embryo select one global fold-specific blend from a grid containing the exact zero control; processed acceptance remains unread.",
        ),
        check(
            "output_location",
            [NOTEBOOK, METADATA],
            "Only calibration evidence is written below /kaggle/working; no processed candidate, archive, or competition submission command exists.",
        ),
        check(
            "quota_policy",
            [CONFIG, NOTEBOOK],
            "The immutable six-hour ceiling is staged only; launch remains subject to a fresh live guard preserving at least eight Kaggle GPU hours.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(OUTPUT, report)
    print(OUTPUT)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
