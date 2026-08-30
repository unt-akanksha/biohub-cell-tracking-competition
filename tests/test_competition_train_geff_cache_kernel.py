from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle/biohub-train-geff-cache-v1"


def test_kernel_is_private_cpu_train_only_packager() -> None:
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text())
    notebook = json.loads((KERNEL / metadata["code_file"]).read_text())
    source = "".join(notebook["cells"][0]["source"])

    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "/competitions/biohub-cell-tracking-during-development/train" in source
    assert "input_root.rglob" not in source
    assert "competition_test_data_read': False" in source
    assert "submission_created': False" in source
    assert "submission.csv" not in source
