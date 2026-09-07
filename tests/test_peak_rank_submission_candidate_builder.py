from __future__ import annotations

import ast
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-peak-rank-submission-candidate.py"
DEPTH_PU_SCRIPT = ROOT / "scripts" / "build-peak-rank-depth-pu-submission-candidate.py"
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
        "dual_association_models_verified",
        "selected_peak_tta_mode",
        "max_worker_elapsed_seconds",
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
    assert joined.count("--peak-tta-mode") == 2
    assert "worker_manifest.json" in joined
    assert "SEC_EDGE_TTA_ACTIVE" in joined
    assert '"source_public_kernel_ref": "redoctopusk/biohub-948tta2"' in joined
    assert "The notebook title's advertised `0.948`" in joined
    assert notebook["metadata"]["codex"]["public_prediction_copied"] is False


def test_depth_pu_candidate_has_distinct_runtime_and_run_identity() -> None:
    wrapper = runpy.run_path(str(DEPTH_PU_SCRIPT), run_name="depth_pu_probe")
    globals_ = wrapper["module"]["main"].__globals__
    notebook = globals_["build_notebook"]("b" * 64)
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert globals_["RUNTIME_REF"].endswith("depth-pu-validation-runtime-v2")
    assert globals_["TARGET_ID"] == "biohub-peak-rank-depth-pu-tracking-candidate-v2"
    assert "peak-rank-depth-pu-tracking-candidate-v2" in joined
    assert '"run_id": "peak-rank-tracking-candidate-v1"' not in joined
