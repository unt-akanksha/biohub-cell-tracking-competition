from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-temporal-contextual-calibration-kernel.py"
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-contextual-calibration-v3"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-contextual-calibration-v3.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"


def build() -> tuple[dict, dict, str]:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, metadata, code


def test_calibration_kernel_is_clean_two_gpu_evidence_only() -> None:
    _notebook, metadata, code = build()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == (
        "indarkarhana/biohub-temporal-contextual-calibration-v3"
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-temporal-contextual-transfer-runtime-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ]
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-temporal-contextual-transfer-v3",
        "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "torch.cuda.device_count() != 2" in code
    assert "DECLARED_BUDGET_SECONDS = 21_600" in code
    assert '"--max-wall-seconds", "18000"' in code
    assert '"--orchestrator-hard-stop-seconds", "19800"' in code
    assert (
        "193478079a0d3f83c1307416c60ed5ad7c74a840fafc5e046ef7f30e2db4b3c1"
        in code
    )
    assert '"--expected-family", "temporal_contextual_pair_fusion_v3"' in code
    assert '"--strict-checkpoint"' in code
    assert '("target_only", 0.0, 0.0)' in code
    assert 'len(row.get("calibration_stems", [])) == 12' in code
    assert "processed_acceptance_ground_truth_read" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "competitions submit" not in code.casefold()
    assert "kaggle competitions" not in code.casefold()


def test_calibration_kernel_builder_is_byte_deterministic() -> None:
    build()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()

    build()

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
