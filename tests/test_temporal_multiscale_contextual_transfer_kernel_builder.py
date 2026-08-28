from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT / "scripts" / "build-temporal-multiscale-contextual-transfer-kernel.py"
)
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-multiscale-transfer-v4"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-multiscale-transfer-v4.ipynb"
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


def test_multiscale_transfer_kernel_is_two_gpu_gated_training_only() -> None:
    _notebook, metadata, code = build()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == "indarkarhana/biohub-temporal-multiscale-transfer-v4"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ]
    assert metadata["kernel_sources"] == [
        "josefreitasalvesneto/biohub-synthetic-dataset",
        "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
        "indarkarhana/biohub-zebrahub-contextual-acceptance-v1",
        "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1",
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "torch.cuda.device_count() != 2" in code
    assert "DECLARED_BUDGET_SECONDS = 39_600" in code
    assert "ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3" in code
    assert "recompute_acceptance_gate" in code
    assert "Acceptance aggregate and child terminal diverge" in code
    assert "Both v4 pretraining checkpoints strict-loaded" in code
    assert "MultiscaleContextualPairFusionAssociationModel" in code
    assert "train_dual_fold_multiscale_contextual_pair_fusion.py" in code
    assert '"--initial-model-root", str(multiscale_pretraining_root)' in code
    assert "temporal_multiscale_contextual_pair_fusion_v4" in code
    assert "hash-bound multiscale ZebraHub external pretraining" in code
    assert 'result.get("both_folds_improved") is True' in code
    assert 'row.get("finetuning_gate_passed") is True' in code
    assert '"--expected-family", "temporal_multiscale_contextual_pair_fusion_v4"' in code
    assert "public_predictions_copied" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "competitions submit" not in code.casefold()
    assert "kaggle competitions" not in code.casefold()


def test_multiscale_transfer_kernel_builder_is_byte_deterministic() -> None:
    build()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()

    build()

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
