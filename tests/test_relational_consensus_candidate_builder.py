from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-relational-consensus-submission-candidate.py"
SPEC = importlib.util.spec_from_file_location("relational_candidate_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_candidate_is_project_authored_additive_and_dual_gpu() -> None:
    assert "project-authored" in module.ATTRIBUTION
    assert "No public prediction" in module.ATTRIBUTION
    assert "At most one" in module.ATTRIBUTION
    assert "Exactly two T4 GPUs are required" in module.MODEL_SETUP_TEMPLATE
    assert 'torch.device("cuda:0")' in module.MODEL_SETUP_TEMPLATE
    assert 'torch.device("cuda:1")' in module.MODEL_SETUP_TEMPLATE
    assert "ThreadPoolExecutor" in module.RANKED_HELPERS
    assert "_RCD_MODELS_BY_DEVICE" in module.RANKED_HELPERS


def test_candidate_uses_tuple_model_without_absolute_threshold() -> None:
    assert "score_relational_candidates" in module.MODEL_SETUP_TEMPLATE
    assert "RelationalDivisionModel" in module.MODEL_SETUP_TEMPLATE
    assert "discover_division_recovery_candidates" in module.MODEL_SETUP_TEMPLATE
    assert '"absolute_threshold_used": False' in module.MODEL_SETUP_TEMPLATE
    assert "model_subset_searched_on_audit" in module.MODEL_SETUP_TEMPLATE
    assert "kaggle competitions submit" not in SCRIPT.read_text(encoding="utf-8")
