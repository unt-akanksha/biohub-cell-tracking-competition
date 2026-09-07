import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify-peak-rank-submission-candidate.py"
SUBMITTER = ROOT / "scripts" / "submit-peak-rank-candidate.py"


def test_verifier_requires_clean_gain_non_regression_and_non_replica() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "PUBLIC_CONTROL_VALIDATOR_SHA256",
        "MINIMUM_PROXY_GAIN = 0.003",
        "MAXIMUM_WEIGHTED_EDGE_REGRESSION = 0.001",
        "MAXIMUM_MOVIE_PROXY_REGRESSION = 0.005",
        "KNOWN_PUBLIC_SUBMISSION_SHA256",
        "SOURCE_PUBLIC_NOTEBOOK_SHA256",
        "secondary_edge_feature_tta",
        "dual_association_models_verified",
        "worker_count",
        "clean_validation_promotion_passed",
        "eligible_for_submission",
        "--expected-run-id",
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
    ):
        assert required in source
