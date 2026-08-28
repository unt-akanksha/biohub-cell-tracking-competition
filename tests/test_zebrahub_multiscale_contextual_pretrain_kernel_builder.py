from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT / "scripts" / "build-zebrahub-multiscale-contextual-pretrain-kernel.py"
)
KERNEL_DIR = ROOT / "kaggle" / "biohub-zebrahub-multiscale-pretrain-v1"
NOTEBOOK = KERNEL_DIR / "biohub-zebrahub-multiscale-pretrain-v1.ipynb"
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


def test_multiscale_pretraining_kernel_is_two_gpu_accepted_v3_only() -> None:
    _notebook, metadata, code = build()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == (
        "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1"
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4",
        "indarkarhana/biohub-zebrahub-contextual-shards-v1",
    ]
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-zebrahub-contextual-pretrain-v1",
        "indarkarhana/biohub-zebrahub-contextual-acceptance-v1",
    ]
    assert metadata["competition_sources"] == []
    assert "torch.cuda.device_count() != 2" in code
    assert "ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3" in code
    assert "recompute_acceptance_gate" in code
    assert "Acceptance aggregate and child terminal diverge" in code
    assert "Both accepted v3 checkpoints strict-loaded" in code
    assert "ContextualPairFusionAssociationModel" in code
    assert "train_zebrahub_multiscale_contextual_pretrain.py" in code
    assert '"--initial-model-root", str(pretraining_root)' in code
    assert '"--patch-batch-size", "32"' in code
    assert '"--gradient-accumulation", "3"' in code
    assert 'row.get("parameter_count") == 46_386_607' in code
    assert "temporal_multiscale_contextual_pair_fusion_v4" in code
    assert "initial_predictions_numerically_preserved" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "competitions submit" not in code.casefold()
    assert "kaggle competitions" not in code.casefold()


def test_multiscale_pretraining_kernel_builder_is_byte_deterministic() -> None:
    build()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()

    build()

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
