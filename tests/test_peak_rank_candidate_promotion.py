import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify-peak-rank-submission-candidate.py"
SUBMITTER = ROOT / "scripts" / "submit-peak-rank-candidate.py"
EXACT_SCORER = ROOT / "research" / "peak_rank_detection" / "score_official_candidate.py"


def test_verifier_requires_patched_official_gain_non_regression_and_non_replica() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "PUBLIC_CONTROL_VALIDATOR_SHA256",
        "MINIMUM_EXACT_SCORE_GAIN = 0.003",
        "MAXIMUM_EXACT_EDGE_REGRESSION = 0.001",
        "MAXIMUM_EXACT_MOVIE_REGRESSION = 0.005",
        "OFFICIAL_SCORER_LOCK_SHA256",
        "official_metric_result.json",
        "public_validator_proxy_used_for_promotion",
        "KNOWN_PUBLIC_SUBMISSION_SHA256",
        "SOURCE_PUBLIC_NOTEBOOK_SHA256",
        "secondary_edge_feature_tta",
        "dual_association_models_verified",
        "worker_count",
        "clean_validation_promotion_passed",
        "eligible_for_submission",
        "--expected-run-id",
        "--official-metric-result",
    ):
        assert required in source


def test_submitter_is_one_shot_and_requires_execute() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "validate_promotion",
        "daily submission limit reached",
        "submission_intent_recorded",
        "--execute",
        "kaggle\", \"competitions\", \"submit",
        "--expected-run-id",
        "exact_score_gain",
        "official_metric_result_sha256",
        "public_validator_proxy_used_for_promotion",
    ):
        assert required in source


def test_exact_scorer_is_frozen_complete_movie_and_non_submitting() -> None:
    source = EXACT_SCORER.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "verify_scorer_lock",
        "EXPECTED_SCORER_LOCK_SHA256",
        "EXPECTED_CONTROL_SCORE = 0.9343483108193262",
        "CONTROL_TREE_SHA256",
        "TRUTH_TREE_SHA256",
        "EXPECTED_CONTROL_COUNTS",
        "MINIMUM_SCORE_GAIN = 0.003",
        "MAXIMUM_MOVIE_SCORE_REGRESSION = 0.005",
        '"44b6_12dfb391"',
        '"44b6_267148e4"',
        '"6bba_062c8d37"',
        '"6bba_07e24132"',
        "mean_score_delta_by_embryo",
        '"competition_submission_performed": False',
    ):
        assert required in source
    assert "kaggle competitions submit" not in source
