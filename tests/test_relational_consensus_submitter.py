from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/submit-relational-consensus-candidate.py"
SPEC = importlib.util.spec_from_file_location("relational_submitter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_submitter_requires_external_promotion_and_exactly_once_receipt() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'promotion.get("status") == "eligible_for_submission"' in text
    assert 'promotion.get("authorized_for_submission") is True' in text
    assert 'float(promotion.get("proxy_gain", 0.0)) >= 0.005' in text
    assert "receipt.exists()" in text
    assert "current_daily_submission_count" in text
    assert '"status": "submission_intent_recorded"' in text
    assert '"competition_submission_performed": True' in text


def test_submitter_preserves_dual_gpu_relational_policy() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "equal_rank_selection_admitted_ensemble" in text
    assert "strongest_selection_individual" in text
    assert "gpu_groups_used_per_movie" in text
    assert module.DAILY_SUBMISSION_LIMIT == 5
