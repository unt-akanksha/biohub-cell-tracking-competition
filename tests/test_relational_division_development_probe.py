from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/temporal_contrastive/score_relational_division_development_probe.py"
SPEC = importlib.util.spec_from_file_location("relational_development_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_probe_pins_exact_inventory_and_strict_audit_policy() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert module.INVENTORY_SHA256 == "be8ff10d3355e2918a3480cb30a4de57c39a94edbb854aba5c9447da02ab30ce"
    assert module.EXPECTED_PARAMETER_COUNT == 48_313_050
    assert "policy_audit_passed" in text
    assert "independently_strong_members" in text
    assert "model_subset_searched_on_audit" in text
    assert '"inference_geometry_eligible_rows") == 9' in text
    assert '"authorized_for_submission": False' in text
    assert "kaggle competitions submit" not in text


def test_probe_geometry_order_matches_training_contract() -> None:
    assert module.GEOMETRY_FIELDS == (
        "parent_distance_um",
        "sister_distance_um",
        "existing_distance_um",
        "daughter_midpoint_distance_um",
        "daughter_opposition_cosine",
        "daughter_step_ratio",
        "biological_geometry_score",
        "parent_velocity_um",
        "constant_velocity_midpoint_error_um",
    )
