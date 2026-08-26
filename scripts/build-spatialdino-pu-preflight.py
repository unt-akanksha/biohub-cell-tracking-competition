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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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


RUN_ID = "spatialdino-pu-adaptation-v1"
KERNEL_DIR = ROOT / "kaggle" / "biohub-spatialdino-pu-adaptation-v1"
NOTEBOOK = KERNEL_DIR / "biohub-spatialdino-pu-adaptation-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
RUNTIME_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-spatialdino-pu-runtime-v1" / "SOURCE_MANIFEST.json"
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
    runtime = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    assertions = {
        NOTEBOOK: "notebook_sha256",
        METADATA: "kernel_metadata_sha256",
        ROOT / "scripts" / "build-spatialdino-pu-adaptation.py": "kernel_builder_sha256",
        ROOT / "scripts" / "build-spatialdino-pu-runtime.py": "runtime_builder_sha256",
        RUNTIME_MANIFEST: "runtime_manifest_sha256",
        GRAPH_MANIFEST: "graph_runtime_manifest_sha256",
        ROOT / "research" / "spatialdino_detection" / "model.py": "model_sha256",
        ROOT / "research" / "spatialdino_detection" / "data.py": "data_sha256",
        ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py": "trainer_sha256",
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
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == []
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert '"--min-steps", "192"' in code
    assert '"--pairs-per-movie", "1"' in code
    assert '"--max-wall-seconds", "5200"' in code
    assert '"--max-wall-seconds", "1300"' in code
    assert "public_predictions_copied" in code
    runtime_root = RUNTIME_MANIFEST.parent
    for name, row in runtime["files"].items():
        assert sha256_file(runtime_root / name) == row["sha256"]

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
    env["PYTHONPATH"] = str(runtime_root)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import data,encoder,model,pu_targets,public_teacher,"
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
            "tests/test_spatialdino_inference.py",
            "tests/test_spatialdino_pu_evaluator.py",
            "tests/test_pu_targets.py",
            "tests/test_public_teacher.py",
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
        default=ROOT / "artifacts" / "preflights" / f"{RUN_ID}-r5.json",
    )
    args = parser.parse_args()
    verify_sources()
    model = ROOT / "research" / "spatialdino_detection" / "model.py"
    data = ROOT / "research" / "spatialdino_detection" / "data.py"
    trainer = ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py"
    inference = ROOT / "research" / "spatialdino_detection" / "inference.py"
    evaluator = ROOT / "research" / "spatialdino_detection" / "evaluate_pu_detector.py"
    checks = [
        check(
            "imports",
            [model, data, trainer, inference, evaluator, NOTEBOOK],
            "All tracked and flat-runtime modules import; notebook cells compile and the focused model/PU/policy suite passes.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME_MANIFEST, GRAPH_MANIFEST, PRIMARY, SECONDARY],
            "Private runtime, teachers, frozen baseline graphs, and competition images are hash-bound with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [model, inference, ROOT / "tests" / "test_spatialdino_detector.py"],
            "The hybrid preserves full spatial resolution, validates tensor geometry, performs flip-consistent inference, and refines peaks sub-voxel.",
        ),
        check(
            "model_step",
            [model, trainer, ROOT / "tests" / "test_spatialdino_pu_trainer.py"],
            "The 29,521,225-parameter architecture has an 8.0M decoder warm-up, a verified 15.1M deep phase, finite forward/backward gradients, an AdamW step, and EMA updates.",
        ),
        check(
            "checkpoint_roundtrip",
            [CHECKPOINT, PRIMARY, SECONDARY, RUNTIME_MANIFEST],
            "SpatialDINO strict-loads at 21,501,312 parameters and both audited teacher checkpoints retain their fixed hashes.",
        ),
        check(
            "output_location",
            [NOTEBOOK, evaluator],
            "Only a learned checkpoint and validation evidence are written under /kaggle/working; submission artifacts cause failure.",
        ),
        check(
            "dense_memory",
            [trainer, inference, CONFIG],
            "Teacher targets and frame pairs are cached on CPU, training is batch-one, validation is batch-four, and soft guards preserve evaluation time.",
        ),
        check(
            "dataset_coverage",
            [data, trainer, evaluator, CONFIG],
            "Every non-validation movie is discovered dynamically; all twelve clean movies are excluded, with eight gating four untouched acceptance movies.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
