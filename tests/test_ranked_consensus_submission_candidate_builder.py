import ast
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-ranked-consensus-submission-candidate.py"


def test_builder_uses_one_project_runtime_and_scale_invariant_policy() -> None:
    module = runpy.run_path(str(BUILDER))

    assert module["TARGET_ID"] == "biohub-ema-ranked-consensus-candidate-v1"
    assert module["CONSENSUS_REF"] == (
        "indarkarhana/biohub-ranked-consensus-division-v1"
    )
    setup = module["MODEL_SETUP_TEMPLATE"]
    helpers = module["RANKED_HELPERS"]
    assert "absolute_threshold_used\") is False" in setup
    assert "Exactly two T4 GPUs are required" in setup
    assert "_RCDDeepModel" in setup
    assert "_RCD_MORPH_PAYLOAD" in setup
    assert '_RCD_INPUT_ROOT = Path("/kaggle/input")' in setup
    assert "predict_proba" in helpers
    assert "apply_ranked_consensus" in helpers
    ast.parse(setup.replace("__MANIFEST_SHA256__", "0" * 64))
    ast.parse(helpers)


def test_candidate_evidence_is_non_submitting_and_hash_bound() -> None:
    module = runpy.run_path(str(BUILDER))
    evidence = module["CANDIDATE_EVIDENCE"]

    assert '"competition_submission_performed": False' in evidence
    assert '"authorized_for_submission": False' in evidence
    assert '"absolute_threshold_used": False' in evidence
    assert '"submission_sha256"' in evidence
    assert "kaggle competitions submit" not in evidence
