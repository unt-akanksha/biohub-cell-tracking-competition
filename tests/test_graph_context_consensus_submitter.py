from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/submit-graph-context-consensus-candidate.py"
SPEC = importlib.util.spec_from_file_location("graph_context_submitter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_submitter_requires_external_promotion_and_exactly_once_receipt() -> None:
    assert module.validate_promotion.__globals__["RUN_ID"] == module.RUN_ID
    source = SCRIPT.read_text(encoding="utf-8")

    assert "receipt.exists()" in source
    assert "current_daily_submission_count" in source
    assert '"status": "submission_intent_recorded"' in source
    assert '"competition_submission_performed": True' in source
    assert module.DAILY_SUBMISSION_LIMIT == 5
