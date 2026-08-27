from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-temporal-patch-blend-kernel.py"
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-patch-dual-fold-blend-v1"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-patch-dual-fold-blend-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"


def test_temporal_patch_blend_kernel_is_strict_two_gpu_calibration_only() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    compile(code, str(NOTEBOOK), "exec")
    assert metadata["id"] == (
        "indarkarhana/biohub-temporal-patch-dual-fold-blend-v1"
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-temporal-patch-runtime-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ]
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-temporal-patch-dual-fold-v1",
        "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "torch.cuda.device_count() != 2" in code
    assert "verify_appearance_output.py" in code
    assert '"--strict-checkpoint"' in code
    assert "verify_trackastra_output.py" in code
    assert '"--max-wall-seconds", "18000"' in code
    assert '"--orchestrator-hard-stop-seconds", "19800"' in code
    assert "DECLARED_BUDGET_SECONDS = 21_600" in code
    assert "processed_acceptance_ground_truth_read" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "competitions submit" not in code
    assert "kaggle competitions" not in code


def test_temporal_patch_blend_kernel_builder_is_byte_deterministic() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], check=True)

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
