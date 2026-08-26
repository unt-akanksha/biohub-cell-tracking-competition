from __future__ import annotations

import argparse
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
from research.lsm_fm_detection.evaluate_ensemble import (  # noqa: E402
    CANDIDATES,
    combine_probabilities,
)
from research.lsm_fm_detection.image_text_model import (  # noqa: E402
    EXPECTED_DETECTOR_PARAMETERS as FEATURE36_PARAMETERS,
    build_lsm_fm_detector as build_feature36,
)
from research.lsm_fm_detection.model import (  # noqa: E402
    EXPECTED_DETECTOR_PARAMETERS as FEATURE24_PARAMETERS,
    build_lsm_fm_detector as build_feature24,
)
from research.spotiflow_biohub.evaluate_pretrained_detector import (  # noqa: E402
    ACCEPTANCE_STEMS,
    SCREEN_STEMS,
)


RUN_ID = "lsm-fm-ensemble-validation-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
RUNTIME = ROOT / ".biohub" / "staging" / "biohub-lsm-fm-ensemble-runtime-v1"
REMOTE = (
    ROOT
    / ".biohub"
    / "cache"
    / "datasets"
    / "biohub-lsm-fm-ensemble-runtime-v1-version1"
)
FEATURE24_OUTPUT = ROOT / ".biohub" / "cache" / "kernel-outputs" / "lsm-fm-pu-adaptation-v2"
FEATURE36_OUTPUT = (
    ROOT
    / ".biohub"
    / "cache"
    / "kernel-outputs"
    / "lsm-fm-image-text-pu-adaptation-v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check(name: str, evidence: list[Path], detail: str, status: str = "passed") -> PreflightCheck:
    return PreflightCheck.create(
        name,
        status,
        [relative(path) for path in evidence],
        ROOT,
        detail=detail,
    )


def verify() -> dict:
    import numpy as np
    import torch

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    assertions = {
        NOTEBOOK: "notebook_sha256",
        METADATA: "kernel_metadata_sha256",
        ROOT / "scripts" / "build-lsm-fm-ensemble-validation.py": "kernel_builder_sha256",
        ROOT / "scripts" / "build-lsm-fm-ensemble-runtime.py": "runtime_builder_sha256",
        RUNTIME / "SOURCE_MANIFEST.json": "runtime_manifest_sha256",
        ROOT / "research" / "lsm_fm_detection" / "evaluate_ensemble.py": "evaluator_sha256",
        ROOT / "research" / "lsm_fm_detection" / "localization_refinement.py": "localization_refinement_sha256",
        ROOT / "research" / "lsm_fm_detection" / "model.py": "feature24_model_code_sha256",
        ROOT / "research" / "lsm_fm_detection" / "image_text_model.py": "feature36_model_code_sha256",
        ROOT / "research" / "spatialdino_detection" / "inference.py": "inference_sha256",
        ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py": "trainer_dependency_sha256",
        ROOT / "research" / "spotiflow_biohub" / "evaluate_pretrained_detector.py": "scorer_sha256",
        ROOT / "research" / "spotiflow_biohub" / "pu_targets.py": "peak_extractor_sha256",
        ROOT / "reports" / "experiments" / "lsm-fm-image-text-pu-adaptation-v1-result.json": "parent_result_report_sha256",
    }
    for path, field in assertions.items():
        assert sha256_file(path) == config[field], (field, path)
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == config["kernel_sources"]
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
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

    manifest = json.loads((RUNTIME / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    remote_mount = REMOTE / manifest["archives"]["monai-1.5.1.zip"]["mount_directory"]
    for name, row in manifest["files"].items():
        assert sha256_file(RUNTIME / name) == row["sha256"], name
        if name == "monai-1.5.1.zip" and remote_mount.is_dir():
            continue
        assert sha256_file(REMOTE / name) == row["sha256"], name
    for name, row in manifest["archives"]["monai-1.5.1.zip"]["files"].items():
        assert sha256_file(remote_mount / name) == row["sha256"], name

    source_paths = {
        "feature24": {
            "launcher": FEATURE24_OUTPUT / "launcher_terminal.json",
            "model": FEATURE24_OUTPUT / "lsm_fm_pu_model" / "best.pt",
            "result": FEATURE24_OUTPUT / "lsm_fm_pu_model" / "pu_training_result.json",
        },
        "feature36": {
            "launcher": FEATURE36_OUTPUT / "launcher_terminal.json",
            "model": FEATURE36_OUTPUT / "lsm_fm_image_text_pu_model" / "best.pt",
            "result": FEATURE36_OUTPUT
            / "lsm_fm_image_text_pu_model"
            / "pu_training_result.json",
        },
    }
    builders = {"feature24": build_feature24, "feature36": build_feature36}
    expected_parameters = {
        "feature24": FEATURE24_PARAMETERS,
        "feature36": FEATURE36_PARAMETERS,
    }
    base_paths = {
        "feature24": REMOTE / "lsm_fm_image_only_student.pt",
        "feature36": REMOTE / "lsm_fm_image_text_student.pt",
    }
    for name, paths in source_paths.items():
        row = config["models"][name]
        assert sha256_file(paths["launcher"]) == row["launcher_terminal_sha256"]
        assert sha256_file(paths["model"]) == row["learned_checkpoint_sha256"]
        assert sha256_file(paths["result"]) == row["training_result_sha256"]
        launcher = json.loads(paths["launcher"].read_text(encoding="utf-8"))
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        assert launcher["status"] == "completed"
        assert launcher["competition_submission_performed"] is False
        assert result["status"] == "completed"
        assert result["validation_overlap"] == []
        assert result["best_weight_sha256"] == row["learned_checkpoint_sha256"]
        model = builders[name](base_paths[name], expected_sha256=row["base_checkpoint_sha256"])
        learned = torch.load(paths["model"], map_location="cpu", weights_only=True)
        model.load_state_dict(learned["state_dict"], strict=True)
        assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters[name]
        del model, learned

    values24 = np.full((4, 4, 4), 0.2, dtype=np.float32)
    values36 = np.full((4, 4, 4), 0.8, dtype=np.float32)
    ensemble = next(value for value in CANDIDATES if value.name == "equal_ensemble_control")
    assert np.allclose(combine_probabilities(values24, values36, ensemble), 0.5)
    assert len(CANDIDATES) == 5
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
                "import evaluate_lsm_fm_ensemble,lsm_fm_image_only_model,"
                "lsm_fm_image_text_model,localization_refinement,monai;"
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
            "tests/test_lsm_fm_ensemble_kernel_builder.py",
            "tests/test_lsm_fm_ensemble_evaluator.py",
            "tests/test_lsm_fm_ensemble_runtime_builder.py",
            "tests/test_lsm_fm_localization_refinement.py",
            "tests/test_lsm_fm_refinement_evaluator.py",
            "tests/test_submission_execution_policy.py",
            "tests/test_submission_sharding.py",
        ],
        cwd=ROOT,
        check=True,
    )
    return source_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json",
    )
    args = parser.parse_args()
    source_paths = verify()
    sources = [path for values in source_paths.values() for path in values.values()]
    remote_manifest = REMOTE / "SOURCE_MANIFEST.json"
    checks = [
        check(
            "imports",
            [
                NOTEBOOK,
                ROOT / "research" / "lsm_fm_detection" / "evaluate_ensemble.py",
                ROOT / "research" / "lsm_fm_detection" / "localization_refinement.py",
                REMOTE / "evaluate_lsm_fm_ensemble.py",
                REMOTE / "monai-1.5.1" / "monai" / "__init__.py",
            ],
            "Notebook cells compile, remote modules import, and focused ensemble/refinement/sharding tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME / "SOURCE_MANIFEST.json", remote_manifest, *sources],
            "Both completed source kernels, learned checkpoints, private runtime, graph baseline, and competition input are exact-hash bound with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [
                ROOT / "research" / "lsm_fm_detection" / "evaluate_ensemble.py",
                ROOT / "tests" / "test_lsm_fm_ensemble_evaluator.py",
                source_paths["feature24"]["model"],
                source_paths["feature36"]["model"],
            ],
            "Both learned detector states strict-load and the fixed equal-probability batch combiner returns finite exact values.",
        ),
        check(
            "model_step",
            [],
            "Inference-only experiment: both source checkpoints already carry completed optimizer evidence and no parameter is updated here.",
            status="not_applicable",
        ),
        check(
            "checkpoint_roundtrip",
            [
                remote_manifest,
                source_paths["feature24"]["model"],
                source_paths["feature24"]["result"],
                source_paths["feature36"]["model"],
                source_paths["feature36"]["result"],
            ],
            "Remote base weights and both learned source checkpoints strict-load with exact expected parameter counts and terminal provenance.",
        ),
        check(
            "output_location",
            [NOTEBOOK, ROOT / "research" / "lsm_fm_detection" / "evaluate_ensemble.py"],
            "Only clean validation evidence is written; submission artifacts cause failure and no competition submission call exists.",
        ),
        check(
            "dataset_coverage",
            [CONFIG, ROOT / "research" / "lsm_fm_detection" / "evaluate_ensemble.py"],
            "One global choice is frozen over eight selection movies before four disjoint acceptance movies can be opened.",
        ),
        check(
            "non_replica_provenance",
            [CONFIG, RUNTIME / "SOURCE_MANIFEST.json", ROOT / "licenses" / "LSM_FM_WEIGHTS_ATTRIBUTION.md"],
            "Only licensed official pretrained weights and independently trained Biohub checkpoints are used; no public predictions or Kaggle code are copied.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
