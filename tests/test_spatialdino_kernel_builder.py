from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-spatialdino-appearance-validation.py"


def _sources() -> tuple[str, dict]:
    values = runpy.run_path(str(BUILDER))
    notebook = {
        "cells": [
            values["code_cell"](values["WATCHDOG"]),
            values["code_cell"](values["SETUP"]),
            values["code_cell"](values["RUN"]),
            values["code_cell"](values["FINISH"]),
        ]
    }
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    return source, values


def test_validation_kernel_is_hash_bound_and_submission_free() -> None:
    source, _values = _sources()

    assert "6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b" in source
    assert "47f199d2e8644ca11be2d5679494bd9607f9e9a7a0b448e35b85391d70c94ed8" in source
    assert 'launcher.get("submission_created") is False' in source
    assert 'Path("/kaggle/working/submission.csv").exists()' in source
    assert "competitions submit" not in source
    assert "association_acceptance_passed" in source


def test_built_metadata_disables_internet_and_tpu(tmp_path, monkeypatch) -> None:
    values = runpy.run_path(str(BUILDER))
    target = tmp_path / "kernel"
    notebook = target / "notebook.ipynb"
    monkeypatch.setitem(values["main"].__globals__, "TARGET", target)
    monkeypatch.setitem(values["main"].__globals__, "NOTEBOOK", notebook)

    values["main"]()

    metadata = json.loads((target / "kernel-metadata.json").read_text())
    built = json.loads(notebook.read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert built["metadata"]["kaggle"]["isInternetEnabled"] is False
