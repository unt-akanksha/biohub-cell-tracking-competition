from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-relational-consensus-division-dataset.py"
SPEC = importlib.util.spec_from_file_location("relational_consensus_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_runtime_inventory_contains_relational_and_independent_voters() -> None:
    assert "relational_division_model.py" in module.RUNTIME_FILES
    assert "relational_division_inference.py" in module.RUNTIME_FILES
    assert "learned_division_recovery.py" in module.RUNTIME_FILES
    assert "handcrafted_division.py" in module.RUNTIME_FILES
    assert module.EXPECTED_PARAMETER_COUNT == 48_313_050


def test_runtime_policy_is_additive_two_gpu_and_submission_ineligible() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"exact_two_t4_required": True' in text
    assert '"maximum_added_edges_per_movie": 1' in text
    assert '"external_policy_additive_only": True' in text
    assert '"model_subset_searched_on_audit": False' in text
    assert '"authorized_for_submission": False' in text
    assert "kaggle competitions submit" not in text
