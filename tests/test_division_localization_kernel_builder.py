from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-division-localization-kernel.py"


def test_localization_kernel_is_offline_two_gpu_and_has_no_submission_path() -> None:
    module = runpy.run_path(str(BUILDER))
    setup = module["SETUP"]
    train = module["TRAIN"]
    combined = "\n".join((module["WATCHDOG"], setup, train, module["FINISH"]))

    assert "len(gpu_lines) != 2" in setup
    assert "T4" in setup
    assert "competition_sources" not in combined
    assert "train_dual_fold_division_localization.py" in train
    assert '"--steps", "6000"' in train
    assert '"--validation-every", "500"' in train
    assert "verify_output(output_root, strict_checkpoint=True)" in train
    assert "kaggle competitions submit" not in combined


def test_built_localization_kernel_metadata_is_exact() -> None:
    module = runpy.run_path(str(BUILDER))
    module["main"]()
    target = ROOT / "kaggle/biohub-multiscale-division-localization-v1"
    metadata = json.loads((target / "kernel-metadata.json").read_text(encoding="ascii"))
    notebook = json.loads(
        (target / "biohub-multiscale-division-localization-v1.ipynb").read_text(
            encoding="ascii"
        )
    )

    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"] == []
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-division-localization-runtime-v1",
        "indarkarhana/biohub-zebrahub-contextual-shards-v1",
        "indarkarhana/biohub-division-localization-shards-v1",
    ]
    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-zebrahub-multiscale-pretrain-v1"
    ]
    assert len(notebook["cells"]) == 5
