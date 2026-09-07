from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-antelume-peak-rank-expanded-real-faint-v7.sh"
GUARD = ROOT / "scripts/run-antelume-biohub-yield-guard-v1.sh"


def test_expanded_real_runner_is_sequential_and_hash_bound() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "while test ! -f \"$v4_run_root/run.complete\" || test ! -f \"$v4_ack\"" in source
    assert "verified_v4_harvest" in source
    assert "sha256sum -c" in source
    assert "competition-real-localization-expanded-shards-v2" in source
    assert 'counts == {"optimization": 480, "selection": 17, "sealed_audit": 14}' in source
    assert "--real-frequency 2" in source
    assert "--widths 128,256,512,1024" in source
    assert "--steps 3000" in source
    assert "--seed 4709011" in source
    assert "--max-wall-seconds 36000" in source


def test_expanded_real_runner_cleanup_cannot_touch_rsna() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "rm -rf" in source
    assert "realpath -m" in source
    assert "/home/ubuntu/biohub-peak-rank-detector-v1/faint-pu-v4/results" in source
    assert "rsna" not in source.lower()
    assert "pkill" not in source
    assert "killall" not in source
    assert "systemctl" not in source


def test_gpu_yield_guard_recognizes_expanded_real_member() -> None:
    source = GUARD.read_text(encoding="utf-8")
    assert "train_expanded_real_faint_detector.py" in source
    assert "kill -s STOP \"$pid\"" in source
    assert "kill -s CONT \"$pid\"" in source
