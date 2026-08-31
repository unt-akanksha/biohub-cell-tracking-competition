from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/restore-antelume-shared-gpu-after-biohub-v1.sh"
)


def test_observer_never_manages_shared_services() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "biohub-temporal-localizer-v2" in source
    assert "biohub-graph-context-recovery-v1" in source
    assert "train_synthetic_localizer.py" in source
    assert "train_graph_context_division_sweep.py" in source
    assert "sleep 15" in source
    assert "shared GPU services were not changed" in source
    assert "systemctl" not in source
    assert "rsna-" not in source
    assert "bigmembers.service" not in source
    assert "plw7.service" not in source
    assert " stop " not in source
    assert " mask " not in source
    assert " unmask " not in source
    assert "rm " not in source
