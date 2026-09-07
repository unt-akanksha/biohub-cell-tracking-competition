import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote-peak-rank-validation-runtime.py"


def test_promoter_requires_clean_hash_bound_non_submission_evidence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        "accepted_for_candidate_integration",
        "validation_result_sha256",
        "checkpoint_sha256",
        "clean_validation_promotion_passed",
        "selected_peak_tta_mode",
        "competition_submission_performed",
        "requires_private_dataset_version",
    ):
        assert required in source
    assert "kaggle competitions submit" not in source
