from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-spatialdino-appearance-validation-v2.py"


def _build(tmp_path: Path) -> tuple[dict, dict]:
    values = runpy.run_path(str(BUILDER))
    target = tmp_path / "kernel"
    notebook = target / "notebook.ipynb"
    values["main"].__globals__["TARGET"] = target
    values["main"].__globals__["NOTEBOOK"] = notebook
    values["main"]()
    return (
        json.loads(notebook.read_text(encoding="ascii")),
        json.loads((target / "kernel-metadata.json").read_text(encoding="ascii")),
    )


def test_v2_uses_hash_bound_dataset_and_keeps_science_unchanged(tmp_path: Path) -> None:
    notebook, _metadata = _build(tmp_path)
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert 'RUN_ID = "spatialdino-appearance-validation-v2"' in code
    assert "/kaggle/input/datasets/indarkarhana/biohub-hoct-processed-validation-v1" in code
    assert "SOURCE_MANIFEST.json" in code
    assert "materialized topology hash mismatch" in code
    assert "ground-truth labels" in code
    assert '"--max-wall-seconds", "6600"' in code
    assert "association_acceptance_passed" in code
    assert "competitions submit" not in code
    assert 'Path("/kaggle/working/submission.csv").exists()' in code


def test_v2_metadata_replaces_kernel_output_with_private_dataset(tmp_path: Path) -> None:
    notebook, metadata = _build(tmp_path)

    assert metadata["kernel_sources"] == []
    assert "indarkarhana/biohub-hoct-processed-validation-v1" in metadata[
        "dataset_sources"
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False

