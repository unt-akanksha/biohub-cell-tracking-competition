from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


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
from research.lsm_fm_detection.center_enhancement import (  # noqa: E402
    build_center_enhancement_model,
)
from research.lsm_fm_detection.evaluate_center_enhancement import (  # noqa: E402
    CANDIDATE_SCALES,
    CONTROL_CANDIDATE,
)
from research.lsm_fm_detection.image_text_model import (  # noqa: E402
    EXPECTED_DETECTOR_PARAMETERS,
    build_lsm_fm_detector,
)
from research.lsm_fm_detection.train_center_enhancement import (  # noqa: E402
    CONTROL_PROBABILITY_POWER,
    CONTROL_REFINEMENT_RADIUS,
    PATCH_SHAPE,
)
from research.spotiflow_biohub.evaluate_pretrained_detector import (  # noqa: E402
    ACCEPTANCE_STEMS,
    SCREEN_STEMS,
)


RUN_ID = "lsm-fm-center-enhancement-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
REMOTE = (
    ROOT
    / ".biohub"
    / "cache"
    / "datasets"
    / "biohub-lsm-fm-center-enhancement-v1-version1"
)
SOURCE_OUTPUT = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "lsm-fm-image-text-pu-adaptation-v1"
)
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}-r3.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check(name: str, evidence: list[Path], detail: str, status: str = "passed"):
    return PreflightCheck.create(
        name,
        status,
        [relative(path) for path in evidence],
        ROOT,
        detail=detail,
    )


def verify() -> dict[str, Path]:
    import torch

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    assertions = {
        ROOT / "scripts" / "build-lsm-fm-center-enhancement.py": "kernel_builder_sha256",
        NOTEBOOK: "notebook_sha256",
        METADATA: "kernel_metadata_sha256",
        REMOTE / "SOURCE_MANIFEST.json": "runtime_manifest_sha256",
        ROOT / "research" / "lsm_fm_detection" / "center_enhancement.py": (
            "center_enhancement_sha256"
        ),
        ROOT / "research" / "lsm_fm_detection" / "train_center_enhancement.py": (
            "trainer_sha256"
        ),
        ROOT / "research" / "lsm_fm_detection" / "evaluate_center_enhancement.py": (
            "evaluator_sha256"
        ),
        ROOT
        / "reports"
        / "experiments"
        / "lsm-fm-image-text-localization-refinement-v1-result.json": (
            "parent_result_report_sha256"
        ),
    }
    for path, field in assertions.items():
        assert sha256_file(path) == config[field], (field, path)
    for name, field in (
        ("center_enhancement.py", "center_enhancement_sha256"),
        ("train_center_enhancement.py", "trainer_sha256"),
        ("evaluate_center_enhancement.py", "evaluator_sha256"),
    ):
        assert sha256_file(REMOTE / name) == config[field]

    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        value.split("@", 1)[0] for value in config["dataset_sources"]
    ]
    assert metadata["kernel_sources"] == config["kernel_sources"]
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    assert "__PENDING" not in code
    assert "competitions submit" not in code
    assert '"--max-wall-seconds", "2400"' in code
    assert '"--max-wall-seconds", "3000"' in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert "train_center_enhancement.py" in code
    assert "evaluate_center_enhancement.py" in code

    paths = {
        "launcher": SOURCE_OUTPUT / "launcher_terminal.json",
        "model": SOURCE_OUTPUT / "lsm_fm_image_text_pu_model" / "best.pt",
        "result": SOURCE_OUTPUT
        / "lsm_fm_image_text_pu_model"
        / "pu_training_result.json",
        "base": REMOTE / "lsm_fm_image_text_student.pt",
    }
    frozen = config["frozen_detector"]
    assert sha256_file(paths["launcher"]) == frozen["launcher_terminal_sha256"]
    assert sha256_file(paths["model"]) == frozen["learned_checkpoint_sha256"]
    assert sha256_file(paths["result"]) == frozen["training_result_sha256"]
    assert sha256_file(paths["base"]) == frozen["base_checkpoint_sha256"]
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["validation_overlap"] == []
    assert result["public_predictions_copied"] is False
    detector = build_lsm_fm_detector(
        paths["base"], expected_sha256=frozen["base_checkpoint_sha256"]
    )
    detector.load_state_dict(
        torch.load(paths["model"], map_location="cpu", weights_only=True)["state_dict"],
        strict=True,
    )
    assert sum(parameter.numel() for parameter in detector.parameters()) == (
        EXPECTED_DETECTOR_PARAMETERS
    )
    refiner = build_center_enhancement_model(channels=32)
    assert sum(parameter.numel() for parameter in refiner.parameters()) == config[
        "center_enhancement"
    ]["parameters"]
    with torch.inference_mode():
        probe = refiner(torch.zeros(2, 2, *PATCH_SHAPE))
    assert probe["logits"].shape == (2, 1, *PATCH_SHAPE)
    assert probe["offsets"].shape == (2, 3)
    assert CONTROL_REFINEMENT_RADIUS == 2
    assert CONTROL_PROBABILITY_POWER == 2.0
    assert CANDIDATE_SCALES == config["candidates"]
    assert CONTROL_CANDIDATE == "feature36_control"
    assert len(SCREEN_STEMS) == 8 and len(ACCEPTANCE_STEMS) == 4
    assert set(SCREEN_STEMS).isdisjoint(ACCEPTANCE_STEMS)

    remote_monai = REMOTE / "monai-1.5.1"
    assert (remote_monai / "monai" / "__init__.py").is_file()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(REMOTE), str(remote_monai)])
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import center_enhancement,train_center_enhancement,"
                "evaluate_center_enhancement,lsm_fm_image_text_model,monai;"
                "assert monai.__version__=='1.5.1'"
            ),
        ],
        cwd=REMOTE,
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_lsm_fm_center_enhancement.py",
            "tests/test_lsm_fm_center_enhancement_training.py",
            "tests/test_lsm_fm_center_enhancement_evaluator.py",
            "tests/test_lsm_fm_center_enhancement_runtime_builder.py",
            "tests/test_lsm_fm_center_enhancement_kernel_builder.py",
            "tests/test_submission_execution_policy.py",
            "tests/test_submission_sharding.py",
        ],
        cwd=ROOT,
        check=True,
    )
    return paths


def main() -> None:
    paths = verify()
    checks = [
        check(
            "imports",
            [
                NOTEBOOK,
                REMOTE / "center_enhancement.py",
                REMOTE / "train_center_enhancement.py",
                REMOTE / "evaluate_center_enhancement.py",
            ],
            "Notebook cells compile, remote modules import, and focused center-enhancement plus two-GPU policy tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, REMOTE / "SOURCE_MANIFEST.json", *paths.values()],
            "The private runtime and frozen feature-36 checkpoint are exact-hash bound; competition input remains attached with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [
                paths["model"],
                ROOT / "tests" / "test_lsm_fm_center_enhancement.py",
            ],
            "The frozen detector strict-loads and the 112,673-parameter refiner completes a two-example 3D forward pass.",
        ),
        check(
            "model_step",
            [
                REMOTE / "train_center_enhancement.py",
                ROOT / "tests" / "test_lsm_fm_center_enhancement_training.py",
            ],
            "Training uses geometry-consistent augmentation, dense center loss, robust residual loss, gradient clipping, and EMA weights.",
        ),
        check(
            "dense_memory",
            [
                CONFIG,
                REMOTE / "train_center_enhancement.py",
                REMOTE / "evaluate_center_enhancement.py",
            ],
            "The 35.1M detector runs with batch size one; 7x7x7 refiner patches use batches of 128 for training and 512 for inference, well within a 16 GB T4.",
        ),
        check(
            "checkpoint_roundtrip",
            [paths["base"], paths["model"], paths["result"], paths["launcher"]],
            "The frozen base and learned detector checkpoint strict-load with exact 35,072,515-parameter provenance; the refiner loader is schema and hash gated.",
        ),
        check(
            "output_location",
            [NOTEBOOK, REMOTE / "evaluate_center_enhancement.py"],
            "Only training and validation evidence is written; submission artifacts cause failure and no submission call exists.",
        ),
        check(
            "dataset_coverage",
            [CONFIG, REMOTE / "data.py", REMOTE / "evaluate_center_enhancement.py"],
            "All twelve validation movies are excluded from sparse-label training; one global scale is chosen on eight movies before four disjoint acceptance movies can open.",
        ),
        check(
            "non_replica_provenance",
            [
                CONFIG,
                ROOT / "licenses" / "LSM_FM_WEIGHTS_ATTRIBUTION.md",
                ROOT
                / "reports"
                / "experiments"
                / "lsm-fm-center-enhancement-design-2026-08-26.md",
            ],
            "The refiner is independently implemented and trained; no CELLECT weights/code, public Kaggle code, public predictions, or leaderboard selection are used.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(OUTPUT, report)
    print(OUTPUT)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
