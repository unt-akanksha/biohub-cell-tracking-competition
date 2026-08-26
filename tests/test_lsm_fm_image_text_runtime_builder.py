from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-image-text-pu-runtime.py"
RUNTIME = ROOT / ".biohub" / "staging" / "biohub-lsm-fm-image-text-pu-runtime-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_image_text_runtime_binds_large_model_and_non_replica_provenance() -> None:
    values = runpy.run_path(str(BUILDER))
    manifest = json.loads((RUNTIME / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert values["checked_target"]() == RUNTIME.resolve()
    assert manifest["pretrained_model"]["license"] == "CC-BY-4.0"
    assert manifest["pretrained_model"]["parameters"] == 36_032_614
    assert manifest["candidate"]["parameters"] == 35_072_515
    assert manifest["candidate"]["public_predictions_copied"] is False
    assert manifest["candidate"]["public_kaggle_code_copied"] is False
    assert manifest["candidate"]["selective_soft_distillation"] is False
    assert len(manifest["archives"]["monai-1.5.1.zip"]["files"]) > 400
    assert manifest["pretrained_model"]["stripped_checkpoint_sha256"] == _sha256(
        RUNTIME / "lsm_fm_image_text_student.pt"
    )
    for name in (
        "lsm_fm_model.py",
        "train_spatialdino_pu_detector.py",
        "evaluate_spatialdino_pu_detector.py",
        "LSM_FM_WEIGHTS_ATTRIBUTION.md",
        "monai-1.5.1.zip",
        "MONAI_LICENSE",
    ):
        assert _sha256(RUNTIME / name) == manifest["files"][name]["sha256"]

