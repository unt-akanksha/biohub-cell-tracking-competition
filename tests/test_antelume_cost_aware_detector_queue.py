from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REWIRE = ROOT / "scripts/rewire-antelume-cost-aware-detector-queue-v1.ps1"
V21 = ROOT / "scripts/run-antelume-peak-rank-expanded-real-xl-balanced-v21.sh"
V27 = ROOT / "scripts/run-antelume-peak-rank-hard-mined-temporal-snr-v27.sh"


def test_rewire_is_narrow_and_preserves_the_active_gpu_owner() -> None:
    source = REWIRE.read_text(encoding="utf-8")
    assert "capacity-pu-v3" in source
    assert "gpu_after[0]" in source
    assert 'test "${gpu_after[0]}" = "${gpu_before[0]}"' in source
    assert "active_v3_interrupted = $false" in source
    assert "rsna_or_unrelated_process_interrupted = $false" in source
    assert "killall" not in source
    assert "pkill" not in source
    assert "shutdown" not in source
    assert "maximum_gpu_hours_removed = 45" in source


def test_high_value_chain_skips_intermediate_ablation_dependencies() -> None:
    v21 = V21.read_text(encoding="utf-8")
    v27 = V27.read_text(encoding="utf-8")
    assert 'while test ! -f "$v3_run_root/run.complete"' in v21
    assert "verified_v3_harvest" in v21
    assert "v19_run_root" not in v21
    assert 'while test ! -f "$v21_run_root/run.complete"' in v27
    assert "verified_v21_harvest" in v27
    assert "v23_run_root" not in v27
