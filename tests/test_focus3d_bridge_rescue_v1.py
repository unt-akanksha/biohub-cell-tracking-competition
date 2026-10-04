from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "scripts/freeze-focus3d-bridge-rescue-v1.py"))


def test_policy_is_deterministic_balanced_and_fail_closed() -> None:
    first = MODULE["freeze"]()
    second = MODULE["freeze"]()
    assert first == second
    stems = first["validation_stems"]
    assert len(stems) == 4 and len(set(stems)) == 4
    assert sum(stem.startswith("44b6_") for stem in stems) == 2
    assert sum(stem.startswith("6bba_") for stem in stems) == 2
    assert not (set(stems) & MODULE["EXCLUDED_STEMS"])
    assert first["validation_selection"]["labels_used"] is False
    assert first["policy"]["maximum_added_nodes_per_movie"] == 1
    assert first["policy"]["maximum_added_edges_per_movie"] == 2
    assert first["authorized_for_submission"] is False
