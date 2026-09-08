from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research/build_graph_context_fresh_split_v3.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fresh_split_v3", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assignment_is_deterministic_disjoint_and_has_frozen_sizes() -> None:
    module = load_module()
    stems = {
        "44b6": [f"44b6_{index:08x}" for index in range(36)],
        "6bba": [f"6bba_{index:08x}" for index in range(110)],
    }
    first = module.assign_roles(stems)
    second = module.assign_roles({key: list(reversed(value)) for key, value in stems.items()})
    assert first == second
    assert len(first) == 146
    assert set(first.values()) == {"optimization", "selection", "audit"}
    for embryo, expected in {
        "44b6": {"audit": 8, "selection": 8, "optimization": 20},
        "6bba": {"audit": 22, "selection": 22, "optimization": 66},
    }.items():
        observed = {
            role: sum(stem.startswith(embryo) and assigned == role for stem, assigned in first.items())
            for role in expected
        }
        assert observed == expected


def test_assignment_rejects_inventory_drift() -> None:
    module = load_module()
    stems = {
        "44b6": [f"44b6_{index:08x}" for index in range(35)],
        "6bba": [f"6bba_{index:08x}" for index in range(110)],
    }
    try:
        module.assign_roles(stems)
    except ValueError as error:
        assert "unexpected 44b6 movie inventory" in str(error)
    else:
        raise AssertionError("inventory drift was accepted")
