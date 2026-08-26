from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PREFLIGHT = ROOT / "scripts" / "build-lsm-fm-pu-preflight.py"
RUN_ID = "lsm-fm-pu-adaptation-v2"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
REPAIR_BUILDER = ROOT / "scripts" / "build-lsm-fm-pu-adaptation-v2.py"
V1_RESULT = ROOT / "reports" / "experiments" / "lsm-fm-pu-adaptation-v1-result.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "preflights" / f"{RUN_ID}.json",
    )
    args = parser.parse_args()

    values = runpy.run_path(str(BASE_PREFLIGHT))
    original_verify = values["verify_sources"]
    shared = original_verify.__globals__
    shared.update(
        {
            "RUN_ID": RUN_ID,
            "KERNEL_DIR": KERNEL_DIR,
            "NOTEBOOK": NOTEBOOK,
            "METADATA": METADATA,
            "CONFIG": CONFIG,
        }
    )
    original_verify()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    v1_result = json.loads(V1_RESULT.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert config["repair_builder_sha256"] == sha256_file(REPAIR_BUILDER)
    assert config["repair_from_v1"]["failure_result_sha256"] == sha256_file(V1_RESULT)
    assert v1_result["optimizer_steps"] == 0
    assert v1_result["scientific_evidence"] is False
    repaired_path = (
        '[str(runtime), str(monai_import_root), str(support_repo.parent), '
        'run_env.get("PYTHONPATH", "")]'
    )
    assert repaired_path in code
    assert code.count('run_env["PYTHONPATH"]') == 1
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_lsm_fm_v2_kernel_builder.py"],
        cwd=ROOT,
        check=True,
    )

    check = values["check"]
    PreflightReport = values["PreflightReport"]
    write_preflight_report = values["write_preflight_report"]
    model = ROOT / "research" / "lsm_fm_detection" / "model.py"
    data = ROOT / "research" / "spatialdino_detection" / "data.py"
    trainer = ROOT / "research" / "spatialdino_detection" / "train_pu_detector.py"
    inference = ROOT / "research" / "spatialdino_detection" / "inference.py"
    evaluator = ROOT / "research" / "spatialdino_detection" / "evaluate_pu_detector.py"
    runtime_manifest = shared["RUNTIME_MANIFEST"]
    graph_manifest = shared["GRAPH_MANIFEST"]
    checkpoint = shared["CHECKPOINT"]
    primary = shared["PRIMARY"]
    secondary = shared["SECONDARY"]
    remote_runtime = shared["REMOTE_RUNTIME"]
    remote_monai_evidence = remote_runtime / "monai-1.5.1" / "monai" / "__init__.py"
    checks = [
        check(
            "imports",
            [model, data, trainer, inference, evaluator, NOTEBOOK, remote_monai_evidence],
            "Tracked and downloaded runtime modules import, the notebook compiles, and all focused tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, runtime_manifest, graph_manifest, checkpoint, primary, secondary],
            "The licensed private runtime, frozen teachers, graph baseline, and competition source remain hash-bound and offline.",
        ),
        check(
            "single_batch",
            [model, checkpoint, ROOT / "tests" / "test_lsm_fm_detector.py"],
            "The exact 64-cubed detector completes a finite CPU forward/backward optimizer step.",
        ),
        check(
            "model_step",
            [model, trainer, CONFIG],
            "The exact 15.7M-parameter warm/deep phases and serialized weak/strong gradients remain unchanged from v1.",
        ),
        check(
            "checkpoint_roundtrip",
            [checkpoint, runtime_manifest, remote_runtime / "lsm_fm_image_only_student.pt"],
            "The stripped LSM-FM student retains its audited hash locally, in staging, and after Kaggle download.",
        ),
        check(
            "output_location",
            [NOTEBOOK, evaluator],
            "Only model and validation evidence can be written; submission files and calls remain absent and guarded.",
        ),
        check(
            "dense_memory",
            [trainer, inference, CONFIG],
            "Batch-one serialized training and batch-one validation retain the v1 memory and wall-time controls.",
        ),
        check(
            "dataset_coverage",
            [data, trainer, evaluator, CONFIG],
            "All 187 training movies and the disjoint eight-plus-four validation protocol remain unchanged.",
        ),
        check(
            "subprocess_environment",
            [REPAIR_BUILDER, NOTEBOOK, V1_RESULT, ROOT / "tests" / "test_lsm_fm_v2_kernel_builder.py"],
            "The only v2 delta prepends the already hash-verified MONAI import root to the shared trainer/evaluator PYTHONPATH; v1 failed before model construction and step zero.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
