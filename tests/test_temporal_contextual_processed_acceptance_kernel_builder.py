from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT / "scripts" / "build-temporal-contextual-processed-acceptance-kernel.py"
)
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-contextual-processed-acceptance-v3"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-contextual-processed-acceptance-v3.ipynb"
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


def test_processed_kernel_is_one_shot_two_gpu_evidence_only() -> None:
    _notebook, metadata, code = build()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == (
        "indarkarhana/biohub-temporal-contextual-processed-acceptance-v3"
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-temporal-contextual-transfer-v3",
        "indarkarhana/biohub-temporal-contextual-calibration-v3",
        "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
        "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v2",
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "torch.cuda.device_count() != 2" in code
    assert "DECLARED_BUDGET_SECONDS = 21_600" in code
    assert '"--hard-stop-seconds", "19800"' in code
    assert (
        "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
        in code
    )
    assert (
        "6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b"
        in code
    )
    assert (
        "559332597da65f161f1b0b116e10fc86c7ff35eb31fe48937e080889b909a43e"
        in code
    )
    assert 'payload.get("both_folds_improved") is True' in code
    assert '"--strict-checkpoint"' in code
    assert 'result.get("whole_movie_sharding") is True' in code
    assert 'result.get("ground_truth_read") is False' in code
    assert 'result.get("exact_processed_scoring_performed") is False' in code
    assert 'result.get("competition_submission_performed") is False' in code
    assert "competitions submit" not in code.casefold()
    assert "kaggle competitions" not in code.casefold()


def test_processed_kernel_builder_is_byte_deterministic() -> None:
    build()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()

    build()

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
