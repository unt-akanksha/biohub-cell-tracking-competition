from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
from research.spotiflow_biohub.public_teacher import load_public_teacher  # noqa: E402


RUN_ID = "spotiflow-pu-adaptation-v1"
KERNEL_DIR = ROOT / "kaggle" / "biohub-spotiflow-pu-adaptation-v1"
NOTEBOOK = KERNEL_DIR / "biohub-spotiflow-pu-adaptation-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
RUNTIME_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-spotiflow-pu-runtime-v1" / "SOURCE_MANIFEST.json"
)
GRAPH_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-trackastra-graph-runtime-v1" / "SOURCE_MANIFEST.json"
)
PRIMARY = (
    ROOT / ".biohub" / "cache" / "datasets" / "teacher-primary" / "edge_predictor_best.pth"
)
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
SPOTIFLOW_MODEL = ROOT / ".biohub" / "cache" / "models" / "spotiflow-0.6.0" / "smfish_3d"
SPOTIFLOW_SMOKE = (
    ROOT / "reports" / "experiments" / "spotiflow-synthetic-finetune-v1-smoke.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def teacher_equivalence() -> None:
    source = ROOT / ".biohub" / "cache" / "datasets" / "biohub-support-source" / "temporal_unet.py"
    spec = importlib.util.spec_from_file_location("audited_temporal_unet", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(PRIMARY, map_location="cpu", weights_only=True)
    upstream = module.TemporalUNet3D(
        in_channels=1, out_channels=32, layers=[32, 64, 128]
    )
    upstream.load_state_dict(
        {key.removeprefix("unet."): value for key, value in state.items() if key.startswith("unet.")}
    )
    head = torch.nn.Conv3d(32, 1, 1)
    head.load_state_dict(
        {
            key.removeprefix("detect_head."): value
            for key, value in state.items()
            if key.startswith("detect_head.")
        }
    )
    upstream.eval()
    head.eval()
    reconstructed = load_public_teacher(PRIMARY)
    generator = torch.Generator().manual_seed(20260826)
    inputs = torch.randn((1, 2, 8, 16, 16), generator=generator)
    with torch.no_grad():
        features = upstream(inputs.unsqueeze(2))
        expected = head(features.reshape(2, 32, 8, 16, 16)).reshape(
            1, 2, 8, 16, 16
        )
        actual = reconstructed(inputs)
    if not torch.equal(expected, actual):
        raise AssertionError(
            f"teacher reconstruction differs from audited source: {(expected - actual).abs().max()}"
        )


def verify_sources() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    runtime = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    assertions = {
        NOTEBOOK: "notebook_sha256",
        METADATA: "kernel_metadata_sha256",
        ROOT / "scripts" / "build-spotiflow-pu-adaptation.py": "kernel_builder_sha256",
        ROOT / "scripts" / "build-spotiflow-pu-runtime.py": "runtime_builder_sha256",
        RUNTIME_MANIFEST: "runtime_manifest_sha256",
        GRAPH_MANIFEST: "graph_runtime_manifest_sha256",
        ROOT / "research" / "spotiflow_biohub" / "pu_targets.py": "target_builder_sha256",
        ROOT / "research" / "spotiflow_biohub" / "public_teacher.py": "teacher_reconstruction_sha256",
        ROOT / "research" / "spotiflow_biohub" / "train_pu_detector.py": "trainer_sha256",
        ROOT / "research" / "spotiflow_biohub" / "evaluate_pu_detector.py": "evaluator_sha256",
    }
    for path, field in assertions.items():
        assert config[field] == sha256_file(path), (field, path)
    assert config["teachers"][0]["checkpoint_sha256"] == sha256_file(PRIMARY)
    assert config["teachers"][1]["checkpoint_sha256"] == sha256_file(SECONDARY)
    assert config["student"]["base_checkpoint_sha256"] == sha256_file(
        SPOTIFLOW_MODEL / "best.pt"
    )
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == []
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert '"--min-steps", "256"' in code
    assert '"--max-wall-seconds", "5200"' in code
    assert '"--max-wall-seconds", "1300"' in code
    runtime_root = RUNTIME_MANIFEST.parent
    for name, expected in runtime["scripts"].items():
        assert sha256_file(runtime_root / name) == expected
    teacher_equivalence()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_pu_targets.py",
            "tests/test_public_teacher.py",
            "tests/test_pu_trainer.py",
            "tests/test_pu_evaluator.py",
            "tests/test_submission_execution_policy.py",
        ],
        cwd=ROOT,
        check=True,
    )
    smoke_payload = json.loads(SPOTIFLOW_SMOKE.read_text(encoding="utf-8"))
    assert smoke_payload["parameter_count"] == 35_489_892
    assert (
        smoke_payload["forward_backward"]["parameter_tensors_with_finite_gradients"]
        > 0
    )
    assert smoke_payload["forward_backward"]["optimizer_step_completed"] is True


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
    targets = ROOT / "research" / "spotiflow_biohub" / "pu_targets.py"
    teacher = ROOT / "research" / "spotiflow_biohub" / "public_teacher.py"
    trainer = ROOT / "research" / "spotiflow_biohub" / "train_pu_detector.py"
    evaluator = ROOT / "research" / "spotiflow_biohub" / "evaluate_pu_detector.py"
    checks = [
        check(
            "imports",
            [targets, teacher, trainer, evaluator, NOTEBOOK],
            "All runtime modules and notebook cells compile; the focused 18-test PU/model/policy suite passes.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME_MANIFEST, GRAPH_MANIFEST, PRIMARY, SECONDARY],
            "The private runtime, public teacher weights, frozen baseline graphs, and competition input are hash-bound with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [targets, ROOT / "tests" / "test_pu_targets.py"],
            "Consensus positives, forced annotations, unknown support, capped safe background, and coordinate-consistent transforms pass deterministic tests.",
        ),
        check(
            "model_step",
            [
                trainer,
                ROOT / "research" / "spotiflow_biohub" / "smoke_model_step.py",
                SPOTIFLOW_SMOKE,
            ],
            "The official 35,489,892-parameter Spotiflow checkpoint completes a finite forward, backward, and AdamW optimizer step on the exact crop shape.",
        ),
        check(
            "checkpoint_roundtrip",
            [teacher, PRIMARY, SECONDARY, SPOTIFLOW_MODEL / "best.pt"],
            "Both teacher checkpoints strict-load into the detection subset, and reconstructed primary logits are bit-identical to the audited public source.",
        ),
        check(
            "output_location",
            [NOTEBOOK, evaluator],
            "Only learned checkpoint and validation evidence are written under /kaggle/working; submission artifacts cause failure and no submit API is present.",
        ),
        check(
            "dense_memory",
            [trainer, CONFIG],
            "Training holds one two-frame crop, frozen teachers, and two student views at a time; a 5,200-second soft stop preserves validation time.",
        ),
        check(
            "dataset_coverage",
            [trainer, evaluator, CONFIG],
            "All twelve selection/acceptance movies are excluded from training; eight selection movies gate access to four untouched acceptance movies.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
