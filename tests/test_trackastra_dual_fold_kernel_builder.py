from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-trackastra-dual-fold-synthetic.py"
TARGET = ROOT / "kaggle" / "biohub-trackastra-dual-fold-synthetic-v1"


def test_builder_emits_strict_two_gpu_training_only_kernel() -> None:
    runpy.run_path(str(BUILDER), run_name="__main__")
    notebook = json.loads(
        (TARGET / "biohub-trackastra-dual-fold-synthetic-v1.ipynb").read_text(
            encoding="ascii"
        )
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    metadata = json.loads((TARGET / "kernel-metadata.json").read_text(encoding="ascii"))

    assert "torch.cuda.device_count() != 2" in code
    assert '"--orchestrate"' in code
    assert '"--real-replay-probability", "0.05"' in code
    assert "kaggle competitions submit" not in code
    assert "submission.zip" not in code
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == ["josefreitasalvesneto/biohub-synthetic-dataset"]
