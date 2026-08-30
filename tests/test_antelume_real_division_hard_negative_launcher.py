from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run-antelume-real-division-hard-negative-v2.sh"


def test_launcher_pins_data_initial_models_and_antelume_gpu() -> None:
    source = SCRIPT.read_text()

    assert "b148a16eef851380c82185b70209e581ac3f7f6c2c2a22362747a0b6d144a605" in source
    assert "bfa974f7c5cfa7b2271b7f65128ac9896d458b3e1af150efb238cc5937a4d251" in source
    assert "c22f716b174654ad06d2c8ca692edc950422488a21bb7b0dafc78c36e042cdab" in source
    assert "b08bbc9795d41ca657910f57dd8a75cd527db98a9f06c880884b8c113c4b6426" in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "nvidia-smi" in source
    assert "A10G" in source
    assert "--steps 4000" in source


def test_launcher_opens_audit_only_after_training_acceptance() -> None:
    source = SCRIPT.read_text()
    training = source.index("train_real_division_gate_v2.py")
    audit = source.index("score_real_division_gate_v2_audit.py")

    assert training < audit
    assert "test ! -e \"$audit_output\"" in source
    assert "kaggle" not in source.lower()
    assert "submission" not in source.lower()
