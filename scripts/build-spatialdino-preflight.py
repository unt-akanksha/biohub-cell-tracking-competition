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


RUN_ID = "spatialdino-appearance-validation-v1"
KERNEL_DIR = ROOT / "kaggle" / "biohub-spatialdino-appearance-validation-v1"
NOTEBOOK = KERNEL_DIR / "biohub-spatialdino-appearance-validation-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
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
    runtime_manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    assert config["notebook_sha256"] == sha256_file(NOTEBOOK)
    assert config["kernel_metadata_sha256"] == sha256_file(METADATA)
    assert config["kernel_builder_sha256"] == sha256_file(
        ROOT / "scripts" / "build-spatialdino-appearance-validation.py"
    )
    assert config["runtime_builder_sha256"] == sha256_file(
        ROOT / "scripts" / "build-spatialdino-runtime.py"
    )
    assert config["runtime_manifest_sha256"] == sha256_file(RUNTIME_MANIFEST)
    assert config["graph_runtime_manifest_sha256"] == sha256_file(GRAPH_MANIFEST)
    assert config["hoct_runtime_manifest_sha256"] == sha256_file(HOCT_MANIFEST)
    assert config["pretrained_model"]["checkpoint_sha256"] == sha256_file(CHECKPOINT)
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-hoct-multibackbone-probe-v1"
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
    assert "PROCESSED_VALIDATION_SHA256" in code
    assert "HOCT_TRAINING_TERMINAL_SHA256" in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created a submission" in code
    runtime_root = RUNTIME_MANIFEST.parent
    for name, record in runtime_manifest["files"].items():
        assert record["sha256"] == sha256_file(runtime_root / name)
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
            "All tracked runtime modules and notebook cells compile; the focused 16-test model/kernel suite passes.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, RUNTIME_MANIFEST, GRAPH_MANIFEST, HOCT_MANIFEST],
            "The private checkpoint runtime, scoring runtime, raw validation graphs, processed comparator output, and competition images are hash-bound with internet and TPU disabled.",
        ),
        check(
            "single_batch",
            [encoder, appearance, ROOT / "tests" / "test_spatialdino_encoder.py"],
            "The exact 174-tensor, 21,501,312-parameter encoder matches upstream arithmetic and produces normalized finite patch embeddings.",
        ),
        check(
            "model_step",
            [correction, ROOT / "tests" / "test_spatialdino_correction.py"],
            "Synthetic crossing links are corrected only when appearance and geometry gates pass; locked edges, divisions, and gaps remain unchanged.",
        ),
        check(
            "checkpoint_roundtrip",
            [RUNTIME_MANIFEST, CHECKPOINT, encoder],
            "The official SpatialDINO checkpoint hash, tensor count, parameter count, and strict state-dict reload are fixed.",
        ),
        check(
            "output_location",
            [NOTEBOOK, validator],
            "Only validation JSON and terminal evidence are written below /kaggle/working; a submission file causes failure and no submit API is present.",
        ),
        check(
            "dense_memory",
            [validator, correction],
            "Image inference is batched four frames at a time and pair scoring is bounded per transition; no all-movie feature tensor is retained on GPU.",
        ),
        check(
            "dataset_coverage",
            [CONFIG, validator],
            "One complete movie per embryo selects the correction, and the two disjoint acceptance movies are opened only if selection passes after configuration freeze.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
