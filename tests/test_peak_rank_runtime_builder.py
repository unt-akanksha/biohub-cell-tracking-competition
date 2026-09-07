import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-peak-rank-validation-runtime.py"


def test_runtime_builder_has_verified_private_inputs_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "accepted_for_kaggle_validation" in source
    assert "audit_passed" in source
    assert "archive_sha256" in source
    assert "checkpoint_sha256" in source
    assert "38_381_478" in source
    assert "public_notebook_weights_read_during_training" in source
    assert "submission.csv" not in source
    assert "kaggle competitions submit" not in source


def test_runtime_builder_packages_inference_and_two_phase_evaluator() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "model.py",
        "inference.py",
        "peak_association_bridge.py",
        "lsm_association_bridge.py",
        "evaluate_peak_rank_detector.py",
        "density_calibration.py",
        "evaluate_pretrained_detector.py",
        "peak_rank_detector.pt",
        "training_terminal.json",
    ):
        assert name in source
    assert "biohub-peak-rank-validation-runtime-v1" in source
