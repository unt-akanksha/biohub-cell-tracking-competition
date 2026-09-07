from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run-antelume-peak-rank-depth-pu-v2.sh"
TRAINER = ROOT / "research" / "peak_rank_detection" / "train_depth_robust_pu_detector.py"


def test_depth_pu_runner_is_precommitted_sequential_and_guard_owned() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "/home/ubuntu/biohub-peak-rank-detector-v1/run.complete" in source
    assert 'sha256sum -c "$predecessor_archive.sha256"' in source
    assert "--steps 3000" in source
    assert "--validation-every 1000" in source
    assert "--seed 1407733" in source
    assert 'cd "$run_root"' in source
    assert 'trainer="$input_root/train_synthetic_real_detector.py"' in source
    assert "while nvidia-smi --query-compute-apps=pid" in source
    for forbidden in ("kill -", "pkill", "killall", "systemctl"):
        assert forbidden not in source


def test_runner_pins_variant_and_base_sources() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "d1b14a8c6c8129cebb71d73020010d7ad41c64b3cc6163b29d758abcc3790dd5" in source
    assert "f64dc9da4b8babab667d3ab216a98759fded2e806032f5c33decc30e31028880" in source
    assert "synthetic256-real-conservative-pu-depth-robust-peak-rank-v2" in source
    assert TRAINER.is_file()
