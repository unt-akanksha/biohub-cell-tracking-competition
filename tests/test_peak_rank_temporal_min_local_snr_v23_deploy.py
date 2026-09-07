from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = (
    ROOT
    / "scripts/deploy-antelume-peak-rank-temporal-min-local-snr-balanced-v23.ps1"
)


def test_deploy_is_hash_bound_and_does_not_control_gpu() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    for digest in (
        "57479c473929aad7f2fdaca5c1322ea88e5f051006faf32c96d066ee4f365a79",
        "dc2c6ed737dfaec2a225c9242343ec76f8aec9e4e5daee2de76f8c4fae501245",
        "7675d0f594fd24172d27d439576d97f46372761f4322de91396f2f05da2a4d5c",
    ):
        assert digest in source
    assert "83812614" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "kaggle competitions submit" not in source


def test_deploy_freezes_temporal_minimum_local_snr_contract() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    assert "TemporalMinimumLocalSnrSafeRankDetector" in source
    assert "temporal_min_local_snr_safe_rank_multiscale_global_peak_rank_v23" in source
    assert "duplicate_44b6_once_balance_embryo_crops" in source
    assert "deployed_waiting_for_v21" in source
