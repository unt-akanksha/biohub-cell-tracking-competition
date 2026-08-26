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


RUN_ID = "hoct-multibackbone-probe-v1"
KERNEL_DIR = ROOT / "kaggle" / "biohub-hoct-multibackbone-probe-v1"
NOTEBOOK = KERNEL_DIR / "biohub-hoct-multibackbone-probe-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"
CONFIG = ROOT / "config" / "experiments" / f"{RUN_ID}.json"
HOCT_MANIFEST = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-hoct-multibackbone-runtime-v1"
    / "SOURCE_MANIFEST.json"
)
GRAPH_MANIFEST = (
    ROOT
    / ".biohub"
    / "staging"
    / "biohub-trackastra-graph-runtime-v1"
    / "SOURCE_MANIFEST.json"
)
MODELS = {
    "general": ROOT / ".biohub" / "cache" / "models" / "hoct" / "general_v1.pt",
    "ctc": ROOT / ".biohub" / "cache" / "models" / "hoct" / "ctc_v0.pt",
}


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
    hoct_manifest = json.loads(HOCT_MANIFEST.read_text(encoding="utf-8"))
    graph_manifest = json.loads(GRAPH_MANIFEST.read_text(encoding="utf-8"))

    assert config["notebook_sha256"] == sha256_file(NOTEBOOK)
    assert config["kernel_metadata_sha256"] == sha256_file(METADATA)
    assert config["kernel_builder_sha256"] == sha256_file(
        ROOT / "scripts" / "build-hoct-multibackbone-probe.py"
    )
    assert config["runtime_manifest_sha256"] == sha256_file(HOCT_MANIFEST)
    assert config["graph_runtime_manifest_sha256"] == sha256_file(GRAPH_MANIFEST)
    assert config["public_validation_materializer_sha256"] == sha256_file(
        ROOT / "research" / "trackastra_graph" / "materialize_public_validation.py"
    )
    assert "indarkarhana/biohub-trackastra-graph-runtime-v1@version11" in config[
        "dataset_sources"
    ]
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v2"
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
    assert "validated_cached_topology" in code
    assert "MAX_REFERENCE_NODE_DRIFT_FRACTION = 0.005" in code
    assert "No valid cached topology found; using the exact in-kernel materializer." in code
    assert "submission.csv" in code

    runtime_root = HOCT_MANIFEST.parent
    for name, record in hoct_manifest["files"].items():
        assert record["sha256"] == sha256_file(runtime_root / name)
    for name, path in MODELS.items():
        assert (
            hoct_manifest["pretrained_models"][name]["sha256"]
            == sha256_file(path)
        )
    graph_root = GRAPH_MANIFEST.parent
    for name in (
        "trainer.py",
        "hybrid_linker.py",
        "rerank_submission.py",
        "materialize_public_validation.py",
        "public_preset_source.py",
        "public_config_source.py",
        "public_postprocess_source.py",
    ):
        assert graph_manifest["files"][name]["sha256"] == sha256_file(
            graph_root / name
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_hoct_kernel_builder.py",
            "tests/test_hoct_probe_trainer.py",
            "tests/test_hoct_multibackbone.py",
            "tests/test_hoct_multibackbone_submission.py",
            "tests/test_hoct_biohub_adapter.py",
        ],
        cwd=ROOT,
        check=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )


def check(
    name: str, evidence: list[Path], detail: str
) -> PreflightCheck:
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

    trainer = ROOT / "research" / "hoct_graph" / "train_biohub_hoct_probe.py"
    multi_trainer = (
        ROOT / "research" / "hoct_graph" / "train_biohub_hoct_multibackbone.py"
    )
    adapter = ROOT / "research" / "hoct_graph" / "biohub_adapter.py"
    grid = ROOT / "research" / "hoct_graph" / "multibackbone.py"
    smoke = ROOT / "reports" / "experiments" / "hoct-probe-local-smoke.json"
    complementarity = (
        ROOT
        / "reports"
        / "experiments"
        / "hoct-backbone-complementarity-audit.json"
    )
    checks = [
        check(
            "imports",
            [trainer, multi_trainer, grid, NOTEBOOK],
            "Both trainers and the twelve-variant ensemble import, every notebook code cell compiles, and the focused suite passes.",
        ),
        check(
            "inputs",
            [CONFIG, METADATA, HOCT_MANIFEST, GRAPH_MANIFEST],
            "The private runtime manifests, exact comparator sources, preceding topology cache, competition data, and offline dependencies are hash-verified with GPU on and internet/TPU off.",
        ),
        check(
            "single_batch",
            [smoke, complementarity, ROOT / "tests" / "test_hoct_biohub_adapter.py"],
            "Both official checkpoints produce finite Biohub edge probabilities; tiled core ownership and parental normalization are covered by tests.",
        ),
        check(
            "model_step",
            [smoke, trainer, ROOT / "tests" / "test_hoct_probe_trainer.py"],
            "A real 288-feature probe step reduces diagnostic BCE, changes head parameters, and the bounded balanced sampler passes deterministic class tests.",
        ),
        check(
            "checkpoint_roundtrip",
            [HOCT_MANIFEST, MODELS["general"], MODELS["ctc"], complementarity],
            "Both official 6,252,593-parameter JIT checkpoints match the staged manifest and reload through the exact inference path.",
        ),
        check(
            "output_location",
            [NOTEBOOK, multi_trainer],
            "Training writes only probes and validation evidence below /kaggle/working, asserts no submission exists, and creates no competition submission.",
        ),
        check(
            "dense_memory",
            [adapter, trainer, ROOT / "tests" / "test_hoct_probe_trainer.py"],
            "Dense inference is spatially tiled, and deterministic per-movie caps hard-bound each backbone to 600,000 float32 feature rows before fitting.",
        ),
        check(
            "dataset_coverage",
            [CONFIG, trainer, multi_trainer],
            "All 195 non-validation movies are scanned, every consecutive transition is covered, candidate recall below 98% aborts, and selection/acceptance stems remain disjoint.",
        ),
    ]
    report = PreflightReport.create(RUN_ID, checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_preflight_report(args.output, report)
    print(args.output)
    print(report.report_sha256)


if __name__ == "__main__":
    main()
