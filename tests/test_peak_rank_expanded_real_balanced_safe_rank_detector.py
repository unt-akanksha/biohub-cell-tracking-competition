from pathlib import Path
from types import SimpleNamespace

from research.peak_rank_detection import (
    train_expanded_real_balanced_safe_rank_detector as balanced,
)


def fake_examples() -> list[SimpleNamespace]:
    return [
        *[SimpleNamespace(identity=f"44b6_movie_{index}") for index in range(150)],
        *[SimpleNamespace(identity=f"6bba_movie_{index}") for index in range(330)],
    ]


def test_only_optimization_role_is_embryo_balanced(monkeypatch) -> None:
    monkeypatch.setattr(
        balanced,
        "_base_load_real_examples",
        lambda paths, *, role: fake_examples(),
    )
    optimization = balanced.load_embryo_balanced_real_examples([], role="optimization")
    selection = balanced.load_embryo_balanced_real_examples([], role="selection")
    assert sum(row.identity.startswith("44b6_") for row in optimization) == 300
    assert sum(row.identity.startswith("6bba_") for row in optimization) == 330
    assert len(selection) == 480
    assert sum(row.identity.startswith("44b6_") for row in selection) == 150


def test_balancer_rejects_changed_optimization_inventory(monkeypatch) -> None:
    monkeypatch.setattr(
        balanced,
        "_base_load_real_examples",
        lambda paths, *, role: fake_examples()[:-1],
    )
    try:
        balanced.load_embryo_balanced_real_examples([], role="optimization")
    except ValueError as error:
        assert "inventory changed" in str(error)
    else:
        raise AssertionError("changed embryo inventory was accepted")


def test_terminal_freezes_balanced_sampling_without_heldout_changes() -> None:
    written = {}

    def writer(path: Path, payload: dict) -> None:
        written[path.name] = payload

    balanced.tagged_atomic_json(
        Path("terminal.json"), {"status": "complete"}, writer=writer
    )
    terminal = written["terminal.json"]
    assert terminal["real_optimization_original_counts"] == {
        "44b6": 150,
        "6bba": 330,
    }
    assert terminal["real_optimization_effective_counts"] == {
        "44b6": 300,
        "6bba": 330,
    }
    assert terminal["selection_sampling_changed"] is False
    assert terminal["sealed_audit_sampling_changed"] is False
    assert terminal["safe_negative_selection_role"] == (
        "expanded_real_optimization_only"
    )
