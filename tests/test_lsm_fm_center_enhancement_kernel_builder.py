from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "lsm-fm-center-enhancement-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"


def test_builder_emits_offline_training_and_selection_gated_kernel() -> None:
    subprocess.run(
        [sys.executable, "scripts/build-lsm-fm-center-enhancement.py"],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text(encoding="ascii"))
    notebook = json.loads(
        (KERNEL_DIR / f"biohub-{RUN_ID}.ipynb").read_text(encoding="ascii")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(KERNEL_DIR), "exec")
    assert metadata["id"] == "indarkarhana/biohub-lsm-fm-center-enhance-v1"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "train_center_enhancement.py" in code
    assert "evaluate_center_enhancement.py" in code
    assert '"--frames-per-movie", "12"' in code
    assert '"--steps", "768"' in code
    assert "selected_candidate" in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert "competitions submit" not in code
    assert "submission.csv" in code
    assert "lsm_fm_center_enhancement.json" in code
