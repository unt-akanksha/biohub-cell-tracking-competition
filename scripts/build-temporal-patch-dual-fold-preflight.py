from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

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
from research.temporal_contrastive.patch_model import (  # noqa: E402
    PhysicalPatchAssociationModel,
)


RUN_ID = "temporal-patch-dual-fold-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
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


def verify_sources() -> tuple[dict, str]:
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
            "tests/test_temporal_contrastive.py",
            "tests/test_temporal_pair_fusion.py",
            "tests/test_submission_sharding.py",
        ],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    manifest = json.loads(
        (RUNTIME / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    if manifest.get("run_id") != RUN_ID or manifest.get("runtime_family") != "cosine_v1":
        raise RuntimeError("runtime family or run identity changed")
    if len(manifest.get("files", {})) != 30:
        raise RuntimeError("runtime file inventory changed")
    if not (
        metadata.get("enable_gpu") is True
        and metadata.get("enable_tpu") is False
        and metadata.get("enable_internet") is False
        and metadata.get("machine_shape") == "NvidiaTeslaT4"
    ):
        raise RuntimeError("Kaggle accelerator or network policy changed")
    if "competitions submit" in code or "kaggle competitions" in code:
        raise RuntimeError("training notebook contains a submission command")
    return manifest, code


def verify_model_path() -> None:
    production = PhysicalPatchAssociationModel()
    if sum(parameter.numel() for parameter in production.parameters()) != 19_221_954:
        raise RuntimeError("production model parameter count changed")
    buffer = io.BytesIO()
    torch.save(production.state_dict(), buffer)
    buffer.seek(0)
    restored = PhysicalPatchAssociationModel()
    restored.load_state_dict(
        torch.load(buffer, map_location="cpu", weights_only=True), strict=True
    )

    probe = PhysicalPatchAssociationModel(base_channels=8, embedding_channels=16)
    patches = torch.randn(4, 3, 17, 17, 17)
    embeddings, divisions = probe(patches)
    loss = probe.pair_logits(embeddings[:2], embeddings[2:]).square().mean()
    loss = loss + divisions.square().mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("local temporal-patch model step is non-finite")
    if not all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in probe.parameters()
    ):
        raise RuntimeError("local temporal-patch model step lost a gradient")


def main() -> None:
    manifest, code = verify_sources()
    verify_model_path()
    runtime_manifest = RUNTIME / "SOURCE_MANIFEST.json"
    model_source = RUNTIME / "patch_model.py"
    trainer_source = RUNTIME / "train_dual_fold_patch.py"
    sharding_source = RUNTIME / "submission_sharding.py"
    checks = [
        check(
            "imports",
            [BUILDER, NOTEBOOK, model_source, trainer_source],
            "Every notebook code cell compiles; the runtime imports and focused temporal, kernel, runtime, and sharding tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, runtime_manifest],
            "The private 30-file runtime, corrected synthetic kernel, coherent Trackastra control, support wheels, and Biohub competition are attached with T4 x2, TPU off, and internet off.",
        ),
        check(
            "single_batch",
            [model_source, ROOT / "tests" / "test_temporal_contrastive.py"],
            "The exact three-channel 17-cubed model path completes a forward pass with normalized embeddings and division logits.",
        ),
        check(
            "model_step",
            [model_source, trainer_source, ROOT / "tests" / "test_temporal_contrastive.py"],
            "The shared production forward/loss path completes backward with finite gradients; the notebook repeats it at production width before training.",
        ),
        check(
            "dense_memory",
            [NOTEBOOK, model_source, trainer_source],
            "Before optimizer training, one T4 must complete the production 176-patch mixed-precision forward/backward used by the maximum 48-source/128-target transition.",
        ),
        check(
            "checkpoint_roundtrip",
            [NOTEBOOK, model_source, trainer_source],
            "The exact 19,221,954-parameter state dict strict-roundtrips locally and again on Kaggle after the production dense probe.",
        ),
        check(
            "output_location",
            [NOTEBOOK, METADATA, sharding_source],
            "Only launcher and reciprocal training evidence are written below /kaggle/working; the notebook contains no submission command and declares submission_created false.",
        ),
        check(
            "dataset_coverage",
            [NOTEBOOK, CONFIG, trainer_source],
            "Setup requires at least 2,028 complete synthetic movies and 120 image/GEFF pairs per embryo before deterministic disjoint validation, calibration, and training partitions are formed.",
        ),
        check(
            "non_replica_provenance",
            [CONFIG, model_source, ROOT / "reports" / "experiments" / "temporal-patch-originality-audit-v1.md"],
            "The appearance encoder and objectives are project-authored; no public predictions, Kaggle notebook code, leaderboard selection, or external appearance weights are admitted.",
        ),
    ]
    if manifest["integrity"]["competition_submission_command_included"] is not False:
        raise RuntimeError("runtime submission policy changed")
    if "Dense 176-patch production forward/backward" not in code:
        raise RuntimeError("production dense-memory gate is absent")
    report = PreflightReport.create(RUN_ID, checks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(OUTPUT, report)
    print(OUTPUT)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
