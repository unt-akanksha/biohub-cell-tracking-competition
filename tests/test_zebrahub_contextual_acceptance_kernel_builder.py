from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-acceptance-kernel.py"
KERNEL_DIR = (
    ROOT / "kaggle" / "biohub-zebrahub-contextual-acceptance-evaluation-v1"
)
NOTEBOOK = (
    KERNEL_DIR / "biohub-zebrahub-contextual-acceptance-evaluation-v1.ipynb"
)
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


def test_acceptance_kernel_is_exactly_two_gpu_and_external_only() -> None:
    _notebook, metadata, code = build()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == (
        "indarkarhana/biohub-zsns001-contextual-gate-v1"
    )
    assert len(metadata["id"].split("/", 1)[1]) <= 50
    assert len(metadata["title"]) <= 50
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-zebrahub-contextual-acceptance-runtime-v1",
        "indarkarhana/biohub-zebrahub-contextual-acceptance-v1",
    ]
    assert metadata["id"] not in metadata["dataset_sources"]
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-zebrahub-contextual-pretrain-v1"
    ]
    assert metadata["competition_sources"] == []
    assert "torch.cuda.device_count() != 2" in code
    assert "DECLARED_BUDGET_SECONDS = 3_600" in code
    assert '"--hard-stop-seconds", "3300"' in code
    assert (
        "6aefc98b953a15b853731bc970a9734ddcab08a51938bcc9769f29fa6bba1f29"
        in code
    )
    assert (
        "cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0"
        in code
    )
    assert 'result.get("gpu_count") == 2' in code
    assert 'result.get("both_folds_improved") is True' in code
    assert 'row.get("gate_passed") is True' in code
    assert 'row.get("selection_or_checkpoint_redirect_permitted") is false' in (
        code.casefold()
    )
    assert "public_predictions_copied" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "kaggle competitions" not in code.casefold()
    assert "competitions submit" not in code.casefold()
    assert "biohub-cell-tracking-during-development" not in code


def test_acceptance_kernel_builder_is_byte_deterministic() -> None:
    build()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()

    build()

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata
