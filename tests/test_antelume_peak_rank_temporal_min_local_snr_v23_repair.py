from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPAIR = (
    ROOT
    / "scripts/repair-antelume-peak-rank-temporal-min-local-snr-v23-scale-contract.sh"
)


def test_repair_is_exactly_hash_bound_and_dormant_only() -> None:
    source = REPAIR.read_text(encoding="utf-8")
    assert "old_model_sha256=dc2c6ed" in source
    assert "old_runner_sha256=7675d0f5" in source
    assert "new_model_sha256=04f2c5d5" in source
    assert "new_runner_sha256=c5948e48" in source
    assert 'test ! -e "$target/results"' in source
    assert 'test ! -e "$target/training.log"' in source
    assert "expanded-real-xl-balanced-v21/run.complete" in source
    assert 'test ! -e /home/ubuntu/biohub-peak-rank-detector-v1/' in source


def test_repair_targets_only_the_exact_v23_waiter() -> None:
    source = REPAIR.read_text(encoding="utf-8")
    assert 'pgrep -f "^bash $target/run.sh$"' in source
    assert 'test "${#pids[@]}" -eq 1' in source
    assert 'kill -TERM "$old_pid"' in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "nvidia-smi" not in source
    assert "RSNA" not in source
    assert "kaggle competitions submit" not in source


def test_repair_preflights_exact_feature_and_parameter_contract() -> None:
    source = REPAIR.read_text(encoding="utf-8")
    assert "LOCAL_SNR_BANDS == ((3, 9), (3, 11), (5, 13))" in source
    assert "83_812_614" in source
    assert 'nohup bash "$runner"' in source
    assert "scale_contract_repaired_waiting_for_v21" in source
