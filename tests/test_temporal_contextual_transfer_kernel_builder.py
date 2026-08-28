from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-temporal-contextual-transfer-kernel.py"
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-contextual-transfer-v3"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-contextual-transfer-v3.ipynb"
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


def test_transfer_kernel_is_two_gpu_gated_training_only() -> None:
    _notebook, metadata, code = build()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == (
        "indarkarhana/biohub-temporal-contextual-transfer-v3"
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
        "josefreitasalvesneto/biohub-synthetic-dataset",
        "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
        "indarkarhana/biohub-zsns001-contextual-gate-v1",
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "torch.cuda.device_count() != 2" in code
    assert "DECLARED_BUDGET_SECONDS = 39_600" in code
    assert '"--steps", "20000"' in code
    assert '"--seed", "41027"' in code
    assert '"--real-replay-probability", "0.60"' in code
    assert '"--learning-rate", "0.00005"' in code
    assert '"--minimum-real-composite-gain", "0.005"' in code
    assert '"--maximum-synthetic-metric-regression", "0.01"' in code
    assert 'result.get("both_folds_improved") is True' in code
    assert 'row.get("finetuning_gate_passed") is True' in code
    assert "recompute_acceptance_gate" in code
    assert "Acceptance aggregate and child terminal diverge" in code
    assert 'launcher.get("acceptance_terminal_sha256")' in code
    assert "for fold in ACCEPTANCE_FOLDS" in code
    assert "Contextual 176-patch/6,144-edge" in code
    assert (
        "cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d"
        in code
    )
    assert "public_predictions_copied" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "competitions submit" not in code.casefold()
    assert "kaggle competitions" not in code.casefold()


def test_transfer_kernel_builder_is_byte_deterministic() -> None:
    build()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()

    build()

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
