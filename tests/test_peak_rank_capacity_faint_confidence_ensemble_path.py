from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/build-peak-rank-capacity-faint-confidence-ensemble-validation-runtime-v6.py"
KERNEL = ROOT / "scripts/build-peak-rank-capacity-faint-confidence-ensemble-validation-kernel-v6.py"
CANDIDATE = ROOT / "scripts/build-peak-rank-capacity-faint-confidence-ensemble-submission-candidate-v6.py"
EVALUATOR = ROOT / "research/peak_rank_detection/evaluate_peak_rank_detector.py"
VERIFIER = ROOT / "scripts/verify-peak-rank-submission-candidate.py"


def test_confidence_runtime_reuses_only_clean_capacity_and_faint_members() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert "build-peak-rank-capacity-faint-ensemble-validation-runtime-v5.py" in source
    assert "confidence_max_logit_with_winner_offset" in source
    assert "clean-capacity-faint-confidence-peak-rank-ensemble-v6" in source
    assert "confidence-selective ensemble" in source

    wrapped = runpy.run_path(str(RUNTIME))
    globals_ = wrapped["module"]["main"].__globals__
    assert globals_["FUSION"] == "confidence_max_logit_with_winner_offset"
    assert globals_["PARAMETER_COUNT"] == 133_955_340


def test_confidence_fusion_is_supported_by_shared_runtime_and_verifier() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    verifier = VERIFIER.read_text(encoding="utf-8")
    assert "PeakRankConfidenceMaxEnsemble" in evaluator
    assert "confidence_max_logit_with_winner_offset" in evaluator
    assert "confidence-selective ensemble" in verifier


def test_confidence_kernel_and_candidate_have_distinct_v6_ids() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    assert "biohub-peak-rank-capacity-faint-confidence-validation-v6" in kernel
    assert '"EXPECTED_PARAMETER_COUNT": 133_955_340' in kernel
    assert "peak-rank-capacity-faint-confidence-tracking-candidate-v6" in candidate
    assert "confidence-selective" in candidate
