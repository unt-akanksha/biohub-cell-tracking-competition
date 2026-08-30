from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-relational-division-development-inventory.py"
SPEC = importlib.util.spec_from_file_location("relational_development_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_candidate_key_accepts_mapping_and_model_record() -> None:
    mapping = {"parent_id": 1, "existing_child_id": 2, "second_child_id": 3}

    class Row:
        parent_id = 1
        existing_child_id = 2
        second_child_id = 3

    assert module.candidate_key(mapping) == (1, 2, 3)
    assert module.candidate_key(Row()) == (1, 2, 3)


def test_inventory_builder_pins_exact_ema_and_never_authorizes_submission() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "artifact_tree_sha256" in text
    assert "SOURCE_INVENTORY_SHA256" in text
    assert '"rows"] == 225' in text
    assert '"safe_recovery_positives"] == 3' in text
    assert '"authorized_for_submission": False' in text
    assert "kaggle competitions submit" not in text
