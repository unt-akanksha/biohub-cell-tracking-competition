from pathlib import Path
import runpy

from tests.test_strong_member_consensus_candidate_promotion import (
    make_output,
    make_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/submit-strong-member-consensus-candidate.py"
MODULE = runpy.run_path(str(SUBMITTER))
VERIFIER = runpy.run_path(
    str(ROOT / "scripts/verify-strong-member-consensus-submission-candidate.py")
)


def test_submitter_accepts_only_unchanged_additive_promotion(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    output, baseline = make_output(tmp_path, runtime)
    promotion = VERIFIER["verify_candidate"](output, baseline, runtime)
    promotion_path = tmp_path / "promotion.json"
    import json

    promotion_path.write_text(json.dumps(promotion), encoding="utf-8")

    payload, submission = MODULE["validate_promotion"](promotion_path)

    assert submission == (output / "submission.csv").resolve()
    assert payload["base_safe_division_heuristic_enabled"] is True
    assert payload["deep_member_count"] == 1


def test_submitter_rejects_removed_base_invariant(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    output, baseline = make_output(tmp_path, runtime)
    promotion = VERIFIER["verify_candidate"](output, baseline, runtime)
    promotion["base_safe_division_heuristic_enabled"] = False
    promotion_path = tmp_path / "promotion.json"
    import json

    promotion_path.write_text(json.dumps(promotion), encoding="utf-8")

    try:
        MODULE["validate_promotion"](promotion_path)
    except RuntimeError as error:
        assert "promotion evidence" in str(error)
    else:
        raise AssertionError("submitter accepted a disabled clean-base invariant")
