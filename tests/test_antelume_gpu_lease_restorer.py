from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/restore-antelume-shared-gpu-after-biohub-v1.sh"
)


def test_restorer_holds_masks_through_localizer_and_graph_recovery() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "biohub-temporal-localizer-v2" in source
    assert "biohub-graph-context-recovery-v1" in source
    assert "train_synthetic_localizer.py" in source
    assert "train_graph_context_division_sweep.py" in source
    assert "systemctl --user unmask bigmembers.service plw7.service" in source
    assert "runtime-masked-rsna-services.txt" in source
    assert "rsna-plw.service rsna-dump.service" in source
    assert "^rsna-[A-Za-z0-9_.@-]+\\.service$" in source
    assert 'sudo -n systemctl stop "$unit"' in source
    assert 'sudo -n systemctl mask --runtime --force "$unit"' in source
    assert 'sudo -n systemctl unmask "$unit"' in source
    assert "sleep 15" in source
    assert "systemctl --user start" not in source
    assert "rm " not in source
