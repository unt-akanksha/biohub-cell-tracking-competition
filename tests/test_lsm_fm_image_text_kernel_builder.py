from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-image-text-pu-adaptation.py"
RUN_ID = "lsm-fm-image-text-pu-adaptation-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = KERNEL_DIR / f"biohub-{RUN_ID}.ipynb"


def test_image_text_kernel_is_heavy_clean_and_submission_sealed() -> None:
    runpy.run_path(str(BUILDER), run_name="__main__")
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    metadata = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(NOTEBOOK), "exec")
    assert "__PENDING" not in code
    assert '"--steps", "3072"' in code
    assert '"--pairs-per-movie", "2"' in code
    assert '"--encoder-blocks", "4"' in code
    assert '"--model-family", "lsm_fm"' in code
    assert '"--batch-size", "1"' in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert "indarkarhana/biohub-lsm-fm-image-text-pu-runtime-v1" in metadata["dataset_sources"]

