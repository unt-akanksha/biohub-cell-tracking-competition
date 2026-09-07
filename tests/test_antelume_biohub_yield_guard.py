from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run-antelume-biohub-yield-guard-v1.sh"
)


def test_guard_only_signals_owned_biohub_gpu_processes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "is_owned_biohub_gpu_pid" in source
    assert "/home/ubuntu/biohub-temporal-localizer-v2/" in source
    assert "/home/ubuntu/biohub-graph-context-recovery-v1/" in source
    assert "/home/ubuntu/biohub-peak-rank-detector-v1/" in source
    assert "/home/ubuntu/biohub)" in source
    assert "train_synthetic_localizer.py" in source
    assert "score_real_development_probe.py" in source
    assert "train_graph_context_division_sweep.py" in source
    assert "score_graph_context_division_development_probe.py" in source
    assert "train_synthetic_real_detector.py" in source
    assert 'kill -s STOP "$pid"' in source
    assert 'kill -s CONT "$pid"' in source
    assert "systemctl" not in source
    assert "pkill" not in source
    assert "killall" not in source


def test_guard_tracks_exact_pids_before_resuming_them() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'record_paused_pid "$pid"' in source
    assert 'done < "$paused_file"' in source
    assert ': > "$paused_file"' in source
    assert 'if is_owned_biohub_gpu_pid "$pid"; then' in source
    assert 'unrelated_gpu_client=true' in source
    assert 'sleep 2' in source
