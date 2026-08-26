from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-spatialdino-pu-distillation-v2.py"
TARGET = ROOT / "kaggle" / "biohub-spatialdino-pu-selective-distillation-v2"


def test_distillation_builder_emits_private_validation_only_kernel() -> None:
    runpy.run_path(str(BUILDER), run_name="__main__")
    metadata = json.loads((TARGET / "kernel-metadata.json").read_text(encoding="ascii"))
    notebook = json.loads(
        (TARGET / "biohub-spatialdino-pu-selective-distillation-v2.ipynb").read_text(
            encoding="ascii"
        )
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert metadata["id"] == (
        "indarkarhana/biohub-spatialdino-pu-selective-distillation-v2"
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == []
    assert "indarkarhana/biohub-spatialdino-pu-runtime-v2" in metadata[
        "dataset_sources"
    ]
    assert '"--distillation-weight", "0.25"' in code
    assert '"--distillation-support-threshold", "0.05"' in code
    assert '"--distillation-agreement-power", "2.0"' in code
    assert "spatialdino-pu-selective-distillation-v2" in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code
    compile(code, str(BUILDER), "exec")
