from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_BUILDER = ROOT / "scripts/build-external-division-policy-runtime.py"
KERNEL_BUILDER = ROOT / "scripts/build-external-division-policy-kernel.py"


def make_runtime(tmp_path: Path) -> Path:
    output = tmp_path / "runtime"
    module = runpy.run_path(str(RUNTIME_BUILDER))
    previous = sys.argv
    try:
        sys.argv = [str(RUNTIME_BUILDER), "--output-root", str(output)]
        module["main"]()
    finally:
        sys.argv = previous
    return output


def test_kernel_is_two_t4_external_only_and_contains_no_submit(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    module = runpy.run_path(str(KERNEL_BUILDER))
    notebook = module["build_notebook"](runtime)
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "Exactly two host T4 GPUs are required" in source
    assert 'environment["CUDA_VISIBLE_DEVICES"] = "0"' in source
    assert "external-division-recovery-policy-v1" in source
    assert "competition_data_read" in source
    assert "kaggle competitions submit" not in source
    assert "submission.csv" not in source


def test_kernel_metadata_attaches_no_competition_source(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    module = runpy.run_path(str(KERNEL_BUILDER))
    target = tmp_path / "kernel"
    target.mkdir()
    target_notebook = target / "kernel.ipynb"
    globals_ = module["main"].__globals__
    globals_["TARGET_DIR"] = target
    globals_["TARGET_NOTEBOOK"] = target_notebook
    previous = sys.argv
    try:
        sys.argv = [str(KERNEL_BUILDER), "--runtime-root", str(runtime)]
        module["main"]()
    finally:
        sys.argv = previous
    metadata = json.loads((target / "kernel-metadata.json").read_text())
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"] == []
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-external-division-policy-runtime-v1",
        "indarkarhana/biohub-zebrahub-contextual-shards-v1",
    ]
