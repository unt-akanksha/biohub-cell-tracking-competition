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
MONAI_DEPS = ROOT / ".biohub" / "cache" / "spatialdino-deps"
for source_root in (MONAI_DEPS, ROOT, ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from biohub_tracker.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)
from research.lsm_fm_detection.image_text_model import (  # noqa: E402
    EXPECTED_DETECTOR_PARAMETERS,
    EXPECTED_PRETRAINED_PARAMETERS,
    build_lsm_fm_detector,
    set_detector_training_phase,
)
from research.spatialdino_detection.data import VALIDATION_STEMS  # noqa: E402
from research.spotiflow_biohub.evaluate_pretrained_detector import (  # noqa: E402
    ACCEPTANCE_STEMS,
    SCREEN_STEMS,
)


RUN_ID = "lsm-fm-image-text-pu-adaptation-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
RUNTIME_ROOT = (
    ROOT / ".biohub" / "staging" / "biohub-lsm-fm-image-text-pu-runtime-v1"
)
RUNTIME_MANIFEST = RUNTIME_ROOT / "SOURCE_MANIFEST.json"
REMOTE_RUNTIME = (
    ROOT
    / ".biohub"
    / "cache"
    / "datasets"
    / "biohub-lsm-fm-image-text-pu-runtime-v1-version1"
)
GRAPH_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-trackastra-graph-runtime-v1" / "SOURCE_MANIFEST.json"
)
CHECKPOINT = (
    ROOT
    / ".biohub"
    / "cache"
    / "models"
    / "lsm-fm"
    / "lsm_fm_image_text_student.pt"
)
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
        ROOT / "scripts" / "build-lsm-fm-image-text-pu-adaptation.py": "kernel_builder_sha256",
        ROOT / "scripts" / "build-lsm-fm-image-text-pu-runtime.py": "runtime_builder_sha256",
        RUNTIME_MANIFEST: "runtime_manifest_sha256",
        GRAPH_MANIFEST: "graph_runtime_manifest_sha256",
        ROOT / "research" / "lsm_fm_detection" / "image_text_model.py": "model_sha256",
        ROOT / "research" / "spatialdino_detection" / "data.py": "data_sha256",
        ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py": "trainer_sha256",
        ROOT / "research" / "spatialdino_detection" / "distillation.py": "distillation_sha256",
        ROOT / "research" / "spatialdino_detection" / "inference.py": "inference_sha256",
        ROOT / "research" / "spatialdino_detection" / "evaluate_pu_detector.py": "evaluator_sha256",
        ROOT / "research" / "spotiflow_biohub" / "pu_targets.py": "target_builder_sha256",
        ROOT / "research" / "spotiflow_biohub" / "public_teacher.py": "teacher_reconstruction_sha256",
        ROOT / "scripts" / "extract-lsm-fm-image-text-student-checkpoint.py": "checkpoint_extractor_sha256",
        ROOT / "licenses" / "LSM_FM_WEIGHTS_ATTRIBUTION.md": "attribution_sha256",
    }
    for path, field in assertions.items():
        assert config[field] == sha256_file(path), (field, path)
    assert config["student"]["stripped_checkpoint_sha256"] == sha256_file(CHECKPOINT)
    assert config["teachers"][0]["checkpoint_sha256"] == sha256_file(PRIMARY)
    assert config["teachers"][1]["checkpoint_sha256"] == sha256_file(SECONDARY)
    assert config["student"]["public_predictions_copied"] is False
    assert config["student"]["public_kaggle_code_copied"] is False
    assert config["student"]["official_finetuning_code_copied"] is False

    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == []
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert "indarkarhana/biohub-lsm-fm-image-text-pu-runtime-v1" in metadata["dataset_sources"]
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    assert "__PENDING" not in code
    assert '"--steps", "3072"' in code
    assert '"--pairs-per-movie", "2"' in code
    assert '"--encoder-blocks", "4"' in code
    assert '"--batch-size", "1"' in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code

    assert manifest["candidate"]["parameters"] == EXPECTED_DETECTOR_PARAMETERS
    assert manifest["candidate"]["public_predictions_copied"] is False
    assert manifest["candidate"]["public_kaggle_code_copied"] is False
    assert manifest["pretrained_model"]["parameters"] == EXPECTED_PRETRAINED_PARAMETERS
    assert manifest["pretrained_model"]["stripped_checkpoint_sha256"] == sha256_file(CHECKPOINT)
    remote_monai_mount = (
        REMOTE_RUNTIME / manifest["archives"]["monai-1.5.1.zip"]["mount_directory"]
    )
    for name, row in manifest["files"].items():
        assert sha256_file(RUNTIME_ROOT / name) == row["sha256"], name
        if name == "monai-1.5.1.zip" and remote_monai_mount.is_dir():
            continue
        assert sha256_file(REMOTE_RUNTIME / name) == row["sha256"], name
    for name, row in manifest["archives"]["monai-1.5.1.zip"]["files"].items():
        assert sha256_file(remote_monai_mount / name) == row["sha256"], name

    model = build_lsm_fm_detector(
        CHECKPOINT,
        expected_sha256=config["student"]["stripped_checkpoint_sha256"],
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 35_072_515
    warm = set_detector_training_phase(model, unfreeze_last_encoder_blocks=0)
    assert warm == {
        "encoder_trainable_parameters": 0,
        "decoder_trainable_parameters": 30_445_381,
        "total_trainable_parameters": 30_445_381,
    }
    torch.manual_seed(20260827)
    image = torch.randn(1, 1, 64, 64, 64)
    output = model(image)
    assert output.shape == image.shape and torch.isfinite(output).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output, torch.zeros_like(output)
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    optimizer.step()
    deep = set_detector_training_phase(model, unfreeze_last_encoder_blocks=4)
    assert deep == {
        "encoder_trainable_parameters": 4_626_810,
        "decoder_trainable_parameters": 30_445_381,
        "total_trainable_parameters": 35_072_191,
    }

    assert len(VALIDATION_STEMS) == 12
    assert len(SCREEN_STEMS) == 8
    assert len(ACCEPTANCE_STEMS) == 4
    assert set(SCREEN_STEMS).isdisjoint(ACCEPTANCE_STEMS)
    assert set(SCREEN_STEMS) | set(ACCEPTANCE_STEMS) == set(VALIDATION_STEMS)

    env = os.environ.copy()
    remote_monai_import = (
        REMOTE_RUNTIME / "monai-1.5.1.zip"
        if (REMOTE_RUNTIME / "monai-1.5.1.zip").is_file()
        else remote_monai_mount
    )
    env["PYTHONPATH"] = os.pathsep.join([str(remote_monai_import), str(REMOTE_RUNTIME)])
    env["BIOHUB_RUNTIME_ROOT"] = str(REMOTE_RUNTIME.resolve())
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os,pathlib,monai,lsm_fm_model,data,distillation,inference,"
                "pu_targets,public_teacher,train_spatialdino_pu_detector,"
                "evaluate_spatialdino_pu_detector;"
                "r=pathlib.Path(os.environ['BIOHUB_RUNTIME_ROOT']);"
                "assert pathlib.Path(monai.__file__).resolve().is_relative_to(r);"
                "m=lsm_fm_model.build_lsm_fm_detector("
                "r/'lsm_fm_image_text_student.pt',expected_sha256='"
                "aca3c5d43ef7f3d7ed2ff169d1ab72b71a03acec293a48283d73d38fcf3520e7');"
                "assert sum(p.numel() for p in m.parameters())==35072515"
            ),
        ],
        cwd=REMOTE_RUNTIME,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_lsm_fm_image_text_detector.py",
            "tests/test_lsm_fm_image_text_kernel_builder.py",
            "tests/test_lsm_fm_image_text_runtime_builder.py",
            "tests/test_spatialdino_pu_trainer.py",
            "tests/test_spatialdino_inference.py",
            "tests/test_spatialdino_pu_evaluator.py",
            "tests/test_pu_targets.py",
            "tests/test_public_teacher.py",
            "tests/test_submission_execution_policy.py",
            "tests/test_submission_sharding.py",
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
    model = ROOT / "research" / "lsm_fm_detection" / "image_text_model.py"
    data = ROOT / "research" / "spatialdino_detection" / "data.py"
    trainer = ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py"
    inference = ROOT / "research" / "spatialdino_detection" / "inference.py"
    evaluator = ROOT / "research" / "spatialdino_detection" / "evaluate_pu_detector.py"
    remote_monai_evidence = REMOTE_RUNTIME / "monai-1.5.1" / "monai" / "__init__.py"
    checks = [
        check(
            "imports",
            [model, data, trainer, inference, evaluator, NOTEBOOK, remote_monai_evidence],
            "Tracked, staged, and independently downloaded runtime modules import; notebook cells compile and focused tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME_MANIFEST, GRAPH_MANIFEST, CHECKPOINT, PRIMARY, SECONDARY],
            "The CC-BY-4.0 weights, Apache-2.0 implementation, private runtime, frozen teachers, graph baseline, and competition source are hash-bound with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [model, CHECKPOINT, ROOT / "tests" / "test_lsm_fm_image_text_detector.py"],
            "An exact 64-cubed CPU forward/backward optimizer step has finite outputs and gradients.",
        ),
        check(
            "model_step",
            [model, trainer, CONFIG],
            "The 35.07M-parameter detector strict-loads all non-head feature-36 tensors and verifies 30.45M warm / 35.07M deep trainable counts.",
        ),
        check(
            "checkpoint_roundtrip",
            [CHECKPOINT, RUNTIME_MANIFEST, REMOTE_RUNTIME / "lsm_fm_image_text_student.pt"],
            "The safely stripped 159-tensor image-text student retains its audited hash locally, in staging, and after Kaggle download.",
        ),
        check(
            "output_location",
            [NOTEBOOK, evaluator],
            "Only learned-model and clean-validation evidence can be written; submission artifacts cause failure and no competition API call exists.",
        ),
        check(
            "dense_memory",
            [trainer, inference, CONFIG],
            "Training serializes weak/strong activation graphs with batch one and CPU target caches; validation remains batch one with hard wall guards.",
        ),
        check(
            "dataset_coverage",
            [data, trainer, evaluator, CONFIG],
            "All 187 training movies use two deterministic pairs; eight selection movies gate four untouched acceptance movies and all twelve are excluded from training.",
        ),
        check(
            "non_replica_provenance",
            [CONFIG, RUNTIME_MANIFEST, ROOT / "licenses" / "LSM_FM_WEIGHTS_ATTRIBUTION.md"],
            "Only licensed pretrained tensors and MONAI are reused; the detector head, PU trainer, evaluator, and outputs are Biohub-owned and no public prediction is copied.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
