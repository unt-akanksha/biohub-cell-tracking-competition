from __future__ import annotations

import ast
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-peak-rank-submission-candidate.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_builder_is_hash_pinned_two_gpu_and_non_submitting() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "SOURCE_NOTEBOOK_SHA256",
        "clean_validation_promotion_passed",
        "predict_with_official_linker.py",
        "torch.cuda.device_count() != 2",
        "peak_worker_manifests",
        '"machine_shape": "NvidiaTeslaT4"',
        "completed_pending_external_promotion_gate",
        "source_advertised_score_used_as_evidence",
        'SOURCE_KERNEL_REF = "redoctopusk/biohub-948tta2"',
        "secondary_edge_feature_tta",
    ):
        assert required in source
    assert "kaggle competitions submit" not in source


def test_transformation_replaces_detector_but_retains_public_linker() -> None:
    notebook = MODULE["build_notebook"]("a" * 64)
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "predict_with_official_linker.py" in joined
    assert joined.count("predict_with_official_linker.py") == 2
    assert "scripts/predict_unet_transformer.py" in joined
    assert "--official-predictor" in joined
    assert "worker_manifest.json" in joined
    assert "SEC_EDGE_TTA_ACTIVE" in joined
    assert '"source_public_kernel_ref": "redoctopusk/biohub-948tta2"' in joined
    assert "The notebook title's advertised `0.948`" in joined
    assert notebook["metadata"]["codex"]["public_prediction_copied"] is False
