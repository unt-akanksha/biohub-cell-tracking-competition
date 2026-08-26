from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-ensemble-validation.py"
RUN_ID = "lsm-fm-ensemble-validation-v1"


def test_ensemble_kernel_is_hash_bound_clean_and_submission_sealed(monkeypatch, tmp_path) -> None:
    kernel_dir = tmp_path / f"biohub-{RUN_ID}"
    notebook_path = kernel_dir / f"biohub-{RUN_ID}.ipynb"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER),
            "--feature36-model-sha256",
            "a" * 64,
            "--feature36-training-result-sha256",
            "b" * 64,
            "--feature36-launcher-sha256",
            "c" * 64,
        ],
    )
    values = runpy.run_path(str(BUILDER))
    main = values["main"]
    main.__globals__.update({"TARGET": kernel_dir, "NOTEBOOK": notebook_path})
    main()
    notebook = json.loads(notebook_path.read_text(encoding="ascii"))
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(notebook_path), "exec")
    assert "a" * 64 in code and "b" * 64 in code and "c" * 64 in code
    assert "__PENDING" not in code
    assert '"--max-wall-seconds", "3000"' in code
    assert '"--batch-size", "1"' in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert "indarkarhana/biohub-lsm-fm-ensemble-runtime-v1" in metadata["dataset_sources"]
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-lsm-fm-pu-adaptation-v2",
        "indarkarhana/biohub-lsm-fm-image-text-pu-adaptation-v1",
    ]
