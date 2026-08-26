from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from research.spatialdino_association.encoder import (  # noqa: E402
    load_spatialdino_vits8,
)
from research.spatialdino_detection.model import (  # noqa: E402
    HybridSpatialDinoDetector,
    set_detector_training_phase,
)


RUN_ID = "spatialdino-pu-selective-distillation-v2"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
RUNTIME_ROOT = ROOT / ".biohub" / "staging" / "biohub-spatialdino-pu-runtime-v2"
RUNTIME_MANIFEST = RUNTIME_ROOT / "SOURCE_MANIFEST.json"
REMOTE_RUNTIME = (
    ROOT
    / ".biohub"
    / "cache"
    / "datasets"
    / "biohub-spatialdino-pu-runtime-v2-version1"
)
GRAPH_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-trackastra-graph-runtime-v1" / "SOURCE_MANIFEST.json"
)
CHECKPOINT = ROOT / ".biohub" / "cache" / "models" / "spatialdino-vits8-step244999-backbone.pth"
PRIMARY = ROOT / ".biohub" / "cache" / "datasets" / "teacher-primary" / "edge_predictor_best.pth"
SECONDARY = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "public-0927-clean-repro-v2"
    / "secondary_seed_weights"
    / "unet_transformer"
    / "split_0"
    / "edge_predictor_best.pth"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def verify_sources() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    assertions = {
        NOTEBOOK: "notebook_sha256",
        METADATA: "kernel_metadata_sha256",
        ROOT / "scripts" / "build-spatialdino-pu-distillation-v2.py": "kernel_builder_sha256",
        ROOT / "scripts" / "build-spatialdino-pu-runtime.py": "runtime_builder_sha256",
        RUNTIME_MANIFEST: "runtime_manifest_sha256",
        GRAPH_MANIFEST: "graph_runtime_manifest_sha256",
        ROOT / "research" / "spatialdino_detection" / "model.py": "model_sha256",
        ROOT / "research" / "spatialdino_detection" / "data.py": "data_sha256",
        ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py": "trainer_sha256",
        ROOT / "research" / "spatialdino_detection" / "distillation.py": "distillation_sha256",
        ROOT / "research" / "spatialdino_detection" / "inference.py": "inference_sha256",
        ROOT / "research" / "spatialdino_detection" / "evaluate_pu_detector.py": "evaluator_sha256",
        ROOT / "research" / "spatialdino_association" / "encoder.py": "encoder_sha256",
        ROOT / "research" / "spotiflow_biohub" / "pu_targets.py": "target_builder_sha256",
        ROOT / "research" / "spotiflow_biohub" / "public_teacher.py": "teacher_reconstruction_sha256",
    }
    for path, field in assertions.items():
        assert config[field] == sha256_file(path), (field, path)
    assert config["student"]["base_checkpoint_sha256"] == sha256_file(CHECKPOINT)
    assert config["teachers"][0]["checkpoint_sha256"] == sha256_file(PRIMARY)
    assert config["teachers"][1]["checkpoint_sha256"] == sha256_file(SECONDARY)
    assert config["training"]["selective_distillation"] == {
        "loss_weight": 0.25,
        "support_threshold": 0.05,
        "agreement_power": 2.0,
        "soft_probability": "mean of the two frozen teacher seeds",
        "soft_weight": "sqrt(primary*secondary) * (1-abs(primary-secondary))^2",
        "unsupported_voxels_weight": 0.0,
    }

    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == []
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert "indarkarhana/biohub-spatialdino-pu-runtime-v2" in metadata["dataset_sources"]
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    assert '"--distillation-weight", "0.25"' in code
    assert '"--distillation-support-threshold", "0.05"' in code
    assert '"--distillation-agreement-power", "2.0"' in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code

    assert manifest["candidate"]["selective_soft_distillation"] is True
    for name, row in manifest["files"].items():
        assert sha256_file(RUNTIME_ROOT / name) == row["sha256"]
        assert sha256_file(REMOTE_RUNTIME / name) == row["sha256"]

    encoder = load_spatialdino_vits8(
        CHECKPOINT,
        expected_sha256=config["student"]["base_checkpoint_sha256"],
    )
    model = HybridSpatialDinoDetector(encoder)
    assert sum(parameter.numel() for parameter in model.parameters()) == 29_521_225
    warm = set_detector_training_phase(model, unfreeze_last_encoder_blocks=0)
    deep = set_detector_training_phase(model, unfreeze_last_encoder_blocks=4)
    assert warm["total_trainable_parameters"] == 8_019_913
    assert deep["total_trainable_parameters"] == 15_121_609

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REMOTE_RUNTIME)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import data,distillation,encoder,model,pu_targets,public_teacher,"
                "train_spatialdino_pu_detector,inference,evaluate_spatialdino_pu_detector;"
                "m=model.HybridSpatialDinoDetector(encoder.SpatialDinoViTS8());"
                "assert sum(p.numel() for p in m.parameters())==29521225"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_spatialdino_encoder.py",
            "tests/test_spatialdino_detector.py",
            "tests/test_spatialdino_pu_trainer.py",
            "tests/test_spatialdino_distillation.py",
            "tests/test_spatialdino_inference.py",
            "tests/test_spatialdino_pu_evaluator.py",
            "tests/test_pu_targets.py",
            "tests/test_public_teacher.py",
            "tests/test_spatialdino_distillation_kernel_builder.py",
            "tests/test_submission_execution_policy.py",
        ],
        cwd=ROOT,
        check=True,
    )


def check(name: str, evidence: list[Path], detail: str) -> PreflightCheck:
    return PreflightCheck.create(
        name,
        "passed",
        [relative(path) for path in evidence],
        ROOT,
        detail=detail,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json",
    )
    args = parser.parse_args()
    verify_sources()
    model = ROOT / "research" / "spatialdino_detection" / "model.py"
    data = ROOT / "research" / "spatialdino_detection" / "data.py"
    trainer = ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py"
    distillation = ROOT / "research" / "spatialdino_detection" / "distillation.py"
    inference = ROOT / "research" / "spatialdino_detection" / "inference.py"
    evaluator = ROOT / "research" / "spatialdino_detection" / "evaluate_pu_detector.py"
    checks = [
        check(
            "imports",
            [model, data, trainer, distillation, inference, evaluator, NOTEBOOK],
            "Tracked, staged, and independently downloaded runtime modules import; notebook cells compile and the focused suite passes.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME_MANIFEST, GRAPH_MANIFEST, PRIMARY, SECONDARY],
            "The private runtime, frozen teachers, baseline graphs, and competition images are hash-bound with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [model, inference, ROOT / "tests" / "test_spatialdino_detector.py"],
            "The full-resolution hybrid has a finite forward/backward optimizer step and sub-voxel inference on a single batch.",
        ),
        check(
            "model_step",
            [model, trainer, distillation, ROOT / "tests" / "test_spatialdino_distillation.py"],
            "The 29.5M-parameter student combines PU, consistency, and differentiable selective soft losses while preserving the verified trainable phases.",
        ),
        check(
            "checkpoint_roundtrip",
            [CHECKPOINT, PRIMARY, SECONDARY, RUNTIME_MANIFEST],
            "SpatialDINO and both frozen teacher checkpoints retain their audited hashes in the downloaded runtime.",
        ),
        check(
            "output_location",
            [NOTEBOOK, evaluator],
            "Only a learned checkpoint and validation evidence are written; submission artifacts cause failure.",
        ),
        check(
            "dense_memory",
            [trainer, inference, CONFIG],
            "Both PU and soft targets share the CPU teacher cache; batch sizes and wall guards preserve evaluation time.",
        ),
        check(
            "dataset_coverage",
            [data, trainer, evaluator, CONFIG],
            "All 187 non-validation movies retain the paired v1 sampling seed; eight selection movies gate four untouched acceptance movies.",
        ),
        check(
            "selective_distillation",
            [distillation, trainer, CONFIG, ROOT / "tests" / "test_spatialdino_distillation.py"],
            "Only two-seed-supported voxels receive soft BCE; unsupported voxels are zero-weight and disagreement is quadratically discounted.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
