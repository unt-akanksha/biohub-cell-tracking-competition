from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-cfe-ensemble-validation-runtime-v8.py"
KERNEL = ROOT / "scripts/build-peak-rank-cfe-ensemble-validation-kernel-v8.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-cfe-ensemble-submission-candidate-v8.py"


def test_triple_runtime_requires_three_individually_gated_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for required in (
        "peak-rank-capacity-pu-validation-controller-v3.json",
        "peak-rank-faint-pu-validation-controller-v4.json",
        "peak-rank-expanded-real-faint-validation-controller-v7.json",
        "member-capacity-pu-v3.pt",
        "member-faint-pu-v4.pt",
        "member-expanded-real-faint-v7.pt",
        "200_933_010",
        "clean-capacity-faint-expanded-equal-logit-ensemble-v8",
    ):
        assert required in source
    assert source.count('"expected_parameter_count": 66_977_670') == 3
    base = (
        ROOT / "scripts/build-peak-rank-logit-ensemble-validation-runtime.py"
    ).read_text(encoding="utf-8")
    assert 'equal_weight = 1.0 / len(members)' in base
    assert 'row["ensemble_weight"] = equal_weight' in base
    assert 'sum(row["ensemble_weight"] for row in members)' in base
    assert '"temporal 3D ConvNeXt U-Net" in architecture' in base
    assert '"public" not in architecture.lower()' in base


def test_triple_validation_and_candidate_have_distinct_identities() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "biohub-peak-rank-cfe-ensemble-validation-v8" in kernel
    assert "biohub-peak-rank-cfe-ensemble-validation-runtime-v8" in kernel
    assert "200_933_010" in kernel
    assert "biohub-peak-rank-cfe-ensemble-validation-runtime-v8" in candidate
    assert "peak-rank-cfe-ensemble-tracking-candidate-v8" in candidate
    assert "200.9M-parameter fixed" in candidate
