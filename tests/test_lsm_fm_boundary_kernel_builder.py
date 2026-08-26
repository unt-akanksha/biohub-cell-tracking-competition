from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "lsm-fm-boundary-refinement-v1"
KERNEL_DIR = ROOT / "kaggle" / f"biohub-{RUN_ID}"


def test_builder_emits_offline_selection_gated_boundary_kernel() -> None:
    subprocess.run(
        [sys.executable, "scripts/build-lsm-fm-boundary-refinement.py"],
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
    assert metadata["id"] == "indarkarhana/biohub-lsm-fm-boundary-refine-v1"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert "evaluate_boundary_refinement.py" in code
    assert '"--max-wall-seconds", "3000"' in code
    assert "selected_strategy" in code
    assert "Validation unexpectedly created submission artifacts" in code
    assert "competitions submit" not in code
    assert "submission.csv" in code
    assert "lsm_fm_boundary_refinement.json" in code
