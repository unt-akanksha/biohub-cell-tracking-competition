from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_BASE = ROOT / "scripts/build-peak-rank-logit-ensemble-validation-runtime.py"
RUNTIME = ROOT / "scripts/build-peak-rank-xl-temporal-snr-pair-validation-runtime-v24.py"
KERNEL = ROOT / "scripts/build-peak-rank-xl-temporal-snr-pair-validation-kernel-v24.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-xl-temporal-snr-pair-submission-candidate-v24.py"


def test_runtime_requires_two_clean_capacity_and_evidence_diverse_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-expanded-real-xl-balanced-validation-controller-v21.json",
        "peak-rank-temporal-min-local-snr-validation-controller-v23.json",
        "member-expanded-real-xl-balanced-v21.pt",
        "member-temporal-min-local-snr-v23.pt",
        "model_temporal_stable.py",
        "213_561_260",
        "clean-xl-temporal-snr-equal-logit-ensemble-v24",
        '"expected_parameter_count": 129_748_646',
        '"expected_parameter_count": 83_812_614',
    ):
        assert required in source


def test_member_architecture_gate_accepts_descriptive_clean_variants() -> None:
    source = ENSEMBLE_BASE.read_text(encoding="utf-8")
    assert '"temporal 3D ConvNeXt U-Net" in architecture' in source
    assert '"public" not in architecture.lower()' in source
    assert 'architecture.startswith("independent ")' not in source


def test_validation_and_candidate_have_distinct_v24_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "xl-temporal-snr-pair-validation-v24" in kernel
    assert "213_561_260" in kernel
    assert "xl-temporal-snr-pair-tracking-candidate-v24" in candidate
    assert "213.6M-parameter fixed XL/temporal-SNR" in candidate
