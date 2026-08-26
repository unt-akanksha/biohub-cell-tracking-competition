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
from research.lsm_fm_detection.evaluate_localization_refinement import (  # noqa: E402
    STRATEGIES,
)
from research.lsm_fm_detection.image_text_model import (  # noqa: E402
    EXPECTED_DETECTOR_PARAMETERS,
    build_lsm_fm_detector,
)
from research.spotiflow_biohub.evaluate_pretrained_detector import (  # noqa: E402
    ACCEPTANCE_STEMS,
    SCREEN_STEMS,
)


RUN_ID = "lsm-fm-image-text-localization-refinement-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
REMOTE = (
    ROOT
    / ".biohub"
    / "cache"
    / "datasets"
    / "biohub-lsm-fm-ensemble-runtime-v1-version1"
)
SOURCE_OUTPUT = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "lsm-fm-image-text-pu-adaptation-v1"
)
OUTPUT = ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check(
    name: str,
    evidence: list[Path],
    detail: str,
    status: str = "passed",
) -> PreflightCheck:
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
        ROOT / "scripts" / "build-lsm-fm-image-text-localization-refinement.py": (
            "kernel_builder_sha256"
        ),
        NOTEBOOK: "notebook_sha256",
        METADATA: "kernel_metadata_sha256",
        REMOTE / "SOURCE_MANIFEST.json": "runtime_manifest_sha256",
        ROOT
        / "research"
        / "lsm_fm_detection"
        / "evaluate_localization_refinement.py": "evaluator_sha256",
        ROOT / "research" / "lsm_fm_detection" / "localization_refinement.py": (
            "localization_refinement_sha256"
        ),
        ROOT
        / "reports"
        / "experiments"
        / "lsm-fm-ensemble-validation-v1-result.json": "parent_result_report_sha256",
    }
    for path, field in assertions.items():
        assert sha256_file(path) == config[field], (field, path)
    assert sha256_file(REMOTE / "evaluate_localization_refinement.py") == config[
        "evaluator_sha256"
    ]
    assert sha256_file(REMOTE / "localization_refinement.py") == config[
        "localization_refinement_sha256"
    ]
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
    assert '"--max-wall-seconds", "3000"' in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert "evaluate_localization_refinement.py" in code

    paths = {
        "launcher": SOURCE_OUTPUT / "launcher_terminal.json",
        "model": SOURCE_OUTPUT / "lsm_fm_image_text_pu_model" / "best.pt",
        "result": SOURCE_OUTPUT
        / "lsm_fm_image_text_pu_model"
        / "pu_training_result.json",
        "base": REMOTE / "lsm_fm_image_text_student.pt",
    }
    model_config = config["model"]
    assert sha256_file(paths["launcher"]) == model_config["launcher_terminal_sha256"]
    assert sha256_file(paths["model"]) == model_config["learned_checkpoint_sha256"]
    assert sha256_file(paths["result"]) == model_config["training_result_sha256"]
    assert sha256_file(paths["base"]) == model_config["base_checkpoint_sha256"]
    launcher = json.loads(paths["launcher"].read_text(encoding="utf-8"))
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert launcher["status"] == "completed"
    assert launcher["competition_submission_performed"] is False
    assert launcher["public_predictions_copied"] is False
    assert result["status"] == "completed"
    assert result["validation_overlap"] == []
    assert result["best_weight_sha256"] == model_config["learned_checkpoint_sha256"]
    model = build_lsm_fm_detector(
        paths["base"], expected_sha256=model_config["base_checkpoint_sha256"]
    )
    learned = torch.load(paths["model"], map_location="cpu", weights_only=True)
    model.load_state_dict(learned["state_dict"], strict=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_DETECTOR_PARAMETERS
    )
    assert [strategy.name for strategy in STRATEGIES] == config["strategies"]
    assert len(SCREEN_STEMS) == 8 and len(ACCEPTANCE_STEMS) == 4
    assert set(SCREEN_STEMS).isdisjoint(ACCEPTANCE_STEMS)

    remote_monai = REMOTE / "monai-1.5.1"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(REMOTE), str(remote_monai)])
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import evaluate_localization_refinement,localization_refinement,"
                "lsm_fm_model,monai;assert monai.__version__=='1.5.1'"
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
            "tests/test_lsm_fm_image_text_refinement_kernel_builder.py",
            "tests/test_lsm_fm_refinement_evaluator.py",
            "tests/test_lsm_fm_localization_refinement.py",
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
                REMOTE / "evaluate_localization_refinement.py",
                REMOTE / "localization_refinement.py",
                REMOTE / "monai-1.5.1" / "monai" / "__init__.py",
            ],
            "Notebook cells compile, remote modules import, and focused refinement and two-GPU policy tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, REMOTE / "SOURCE_MANIFEST.json", *paths.values()],
            "The private runtime and completed feature-36 checkpoint are exact-hash bound; competition input remains attached with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [
                paths["model"],
                ROOT / "tests" / "test_lsm_fm_localization_refinement.py",
            ],
            "The learned state strict-loads and synthetic sub-voxel tests cover every refinement implementation.",
        ),
        check(
            "model_step",
            [],
            "Inference-only experiment: no parameter update occurs.",
            status="not_applicable",
        ),
        check(
            "checkpoint_roundtrip",
            [paths["base"], paths["model"], paths["result"], paths["launcher"]],
            "The base and learned checkpoints strict-load with exact 35,072,515 parameters and clean terminal provenance.",
        ),
        check(
            "output_location",
            [NOTEBOOK, REMOTE / "evaluate_localization_refinement.py"],
            "Only validation evidence is written; submission artifacts cause failure and no submission call exists.",
        ),
        check(
            "dataset_coverage",
            [CONFIG, REMOTE / "evaluate_localization_refinement.py"],
            "One global strategy is selected on eight movies before four disjoint acceptance movies can open.",
        ),
        check(
            "non_replica_provenance",
            [
                CONFIG,
                ROOT / "licenses" / "LSM_FM_WEIGHTS_ATTRIBUTION.md",
                ROOT
                / "reports"
                / "experiments"
                / "lsm-fm-ensemble-validation-v1-result.json",
            ],
            "Only our independently trained feature-36 checkpoint and raw images are used; no public predictions or Kaggle code are copied.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(OUTPUT, report)
    print(OUTPUT)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
