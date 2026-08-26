from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from biohub_tracker.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightReport,
    write_preflight_report,
)


RUN_ID = "spatialdino-appearance-validation-v2"
KERNEL_DIR = ROOT / "kaggle" / "biohub-spatialdino-appearance-validation-v2"
NOTEBOOK = KERNEL_DIR / "biohub-spatialdino-appearance-validation-v2.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
KERNEL_BUILDER = ROOT / "scripts" / "build-spatialdino-appearance-validation-v2.py"
BASE_BUILDER = ROOT / "scripts" / "build-spatialdino-appearance-validation.py"
TOPOLOGY_BUILDER = ROOT / "scripts" / "build-hoct-processed-validation-dataset.py"
TOPOLOGY_ROOT = ROOT / ".biohub" / "staging" / "biohub-hoct-processed-validation-v1"
TOPOLOGY_MANIFEST = TOPOLOGY_ROOT / "SOURCE_MANIFEST.json"
RUNTIME_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-spatialdino-runtime-v1" / "SOURCE_MANIFEST.json"
)
GRAPH_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-trackastra-graph-runtime-v1" / "SOURCE_MANIFEST.json"
)
HOCT_MANIFEST = (
    ROOT / ".biohub" / "staging" / "biohub-hoct-multibackbone-runtime-v1" / "SOURCE_MANIFEST.json"
)
CHECKPOINT = ROOT / ".biohub" / "cache" / "models" / "spatialdino-vits8-step244999-backbone.pth"


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
    topology_manifest = json.loads(TOPOLOGY_MANIFEST.read_text(encoding="utf-8"))

    assert config["notebook_sha256"] == sha256_file(NOTEBOOK)
    assert config["kernel_metadata_sha256"] == sha256_file(METADATA)
    assert config["kernel_builder_sha256"] == sha256_file(KERNEL_BUILDER)
    assert config["scientific_base_builder_sha256"] == sha256_file(BASE_BUILDER)
    assert config["topology_dataset_builder_sha256"] == sha256_file(TOPOLOGY_BUILDER)
    assert config["topology_dataset_manifest_sha256"] == sha256_file(TOPOLOGY_MANIFEST)
    assert config["runtime_manifest_sha256"] == sha256_file(RUNTIME_MANIFEST)
    assert config["graph_runtime_manifest_sha256"] == sha256_file(GRAPH_MANIFEST)
    assert config["hoct_runtime_manifest_sha256"] == sha256_file(HOCT_MANIFEST)
    assert config["pretrained_model"]["checkpoint_sha256"] == sha256_file(CHECKPOINT)

    assert topology_manifest["producer_run_id"] == "hoct-multibackbone-probe-v1"
    assert topology_manifest["producer_status"] == "completed"
    assert topology_manifest["producer_submission_created"] is False
    assert topology_manifest["ground_truth_labels_in_processed_csv"] is False
    for name, record in topology_manifest["files"].items():
        assert record["sha256"] == sha256_file(TOPOLOGY_ROOT / name)

    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == []
    assert "indarkarhana/biohub-hoct-processed-validation-v1" in metadata[
        "dataset_sources"
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]

    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    assert "SOURCE_MANIFEST.json" in code
    assert "materialized topology hash mismatch" in code
    assert "ground-truth labels" in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created a submission" in code

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_spatialdino_encoder.py",
            "tests/test_spatialdino_appearance.py",
            "tests/test_spatialdino_correction.py",
            "tests/test_spatialdino_validation.py",
            "tests/test_spatialdino_kernel_builder.py",
            "tests/test_spatialdino_appearance_v2_kernel_builder.py",
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
    encoder = ROOT / "research" / "spatialdino_association" / "encoder.py"
    appearance = ROOT / "research" / "spatialdino_association" / "appearance.py"
    correction = ROOT / "research" / "spatialdino_association" / "correction.py"
    validator = (
        ROOT
        / "research"
        / "spatialdino_association"
        / "validate_appearance_correction.py"
    )
    checks = [
        check(
            "imports",
            [encoder, appearance, correction, validator, NOTEBOOK],
            "All runtime modules and notebook cells compile; the focused model and kernel tests pass.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME_MANIFEST, GRAPH_MANIFEST, HOCT_MANIFEST],
            "The private runtimes, competition images, and processed topology dataset are explicitly attached with internet and TPU disabled.",
        ),
        check(
            "materialized_topology",
            [TOPOLOGY_BUILDER, TOPOLOGY_MANIFEST],
            "The repaired input is the exact hash-bound output of the completed clean HOCT producer and excludes ground-truth labels.",
        ),
        check(
            "model_step",
            [encoder, appearance, correction],
            "Frozen SpatialDINO appearance can only propose degree-preserving swaps through the unchanged 18-configuration grid.",
        ),
        check(
            "checkpoint_roundtrip",
            [RUNTIME_MANIFEST, CHECKPOINT, encoder],
            "The public MIT-licensed SpatialDINO checkpoint and dependency-light encoder are hash-bound.",
        ),
        check(
            "output_location",
            [NOTEBOOK, validator],
            "Only validation evidence is written below /kaggle/working; submission creation is forbidden.",
        ),
        check(
            "dense_memory",
            [validator, correction],
            "Inference is frame-batched and transition-local, with a 6,900-second hard stop inside the two-hour declaration.",
        ),
        check(
            "dataset_coverage",
            [CONFIG, validator],
            "Two complete selection movies freeze configuration before two disjoint acceptance movies are opened.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
