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


RUN_ID = "temporal-patch-dual-fold-v1-runtime-mount-retry"
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-patch-dual-fold-v1"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-patch-dual-fold-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
BUILDER = ROOT / "scripts" / "build-temporal-patch-dual-fold-kernel.py"
RUNTIME = ROOT / ".biohub" / "staging" / "biohub-temporal-patch-runtime-v1"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}-standard.json"


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
    required = (
        "def materialize_runtime_input",
        "shutil.copytree(source, runtime / source.name)",
        "Ambiguous runtime directory and archive",
        "Dense 176-patch production forward/backward",
        "torch.cuda.device_count() != 2",
    )
    if any(item not in code for item in required):
        raise RuntimeError("retry notebook is missing a required repair or GPU gate")
    if not (
        manifest.get("run_id") == "temporal-patch-dual-fold-v1"
        and len(manifest.get("files", {})) == 30
        and metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
        and config.get("scientific_configuration_changed") is False
        and config.get("required_visible_gpu_count") == 2
        and config.get("authorized_for_submission") is False
    ):
        raise RuntimeError("retry input, accelerator, or scientific policy changed")
    if "competitions submit" in code or "kaggle competitions" in code:
        raise RuntimeError("retry notebook contains a submission command")

    runtime_manifest = RUNTIME / "SOURCE_MANIFEST.json"
    test_path = ROOT / "tests" / "test_temporal_patch_kernel_builder.py"
    checks = [
        check(
            "imports",
            [BUILDER, NOTEBOOK, test_path],
            "Every notebook cell compiles and focused runtime, training, output-verifier, mount-representation, and sharding tests pass.",
        ),
        check(
            "inputs",
            [BUILDER, NOTEBOOK, test_path],
            "The private 30-file runtime, corrected synthetic kernel, coherent Trackastra control, support wheels, and Biohub competition are attached; the exact materializer passes directory and ZIP mount regression tests.",
        ),
        check(
            "single_batch",
            [RUNTIME / "patch_model.py", ROOT / "tests" / "test_temporal_contrastive.py"],
            "The unchanged three-channel 17-cubed appearance model completes the focused forward-path contract.",
        ),
        check(
            "model_step",
            [RUNTIME / "patch_model.py", RUNTIME / "train_dual_fold_patch.py", test_path],
            "The unchanged production forward/loss/backward path and mount-aware setup tests pass with finite gradients.",
        ),
        check(
            "checkpoint_roundtrip",
            [NOTEBOOK, RUNTIME / "patch_model.py"],
            "The strict 19,221,954-parameter checkpoint roundtrip remains required before optimizer training.",
        ),
        check(
            "output_location",
            [NOTEBOOK, METADATA],
            "The retry writes training and launcher evidence only; it contains no competition artifact or submission command.",
        ),
        check(
            "dataset_coverage",
            [CONFIG, runtime_manifest],
            "The unchanged 1,900/128 synthetic and 96/12/12 reciprocal real partitions retain all coverage and opened-acceptance exclusions.",
        ),
        check(
            "dense_memory",
            [NOTEBOOK, RUNTIME / "patch_model.py"],
            "Before optimizer training, each T4 path still requires the production 176-patch mixed-precision forward/backward and strict 19,221,954-parameter checkpoint roundtrip.",
        ),
        check(
            "failure_provenance",
            [CONFIG],
            "The 11.964-second pre-training mount failure and both downloaded evidence hashes are bound; it is not treated as a scientific model result.",
        ),
        check(
            "quota_policy",
            [CONFIG, NOTEBOOK],
            "The retry retains an immutable 11-hour ceiling and requires a fresh live guard projecting at least eight Kaggle GPU hours remaining.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(OUTPUT, report)
    print(OUTPUT)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
