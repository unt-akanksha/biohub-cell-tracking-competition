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
        "04f2c5d580279b7d035007ef0d5c422ad6ee43fec430f8f8a809d69858db172e",
        "c5948e48cfb0c07180cd1927eccaadc1540015a09d7d8543ec70b1f26ce88e7f",
    ):
        assert digest in source
    assert "83812614" in source
    assert "ln -s 'research/peak_rank_detection/model.py'" in source
    assert "readlink -f '/home/ubuntu/biohub/model.py'" in source
    assert "nohup bash" in source
    assert "nvidia-smi" not in source
    assert "pkill" not in source
    assert "killall" not in source
    assert "kaggle competitions submit" not in source


def test_deploy_freezes_temporal_minimum_local_snr_contract() -> None:
    source = DEPLOY.read_text(encoding="utf-8")
    assert "TemporalMinimumLocalSnrSafeRankDetector" in source
    assert "LOCAL_SNR_BANDS == ((3, 9), (3, 11), (5, 13))" in source
    assert "temporal_min_local_snr_safe_rank_multiscale_global_peak_rank_v23" in source
    assert "duplicate_44b6_once_balance_embryo_crops" in source
    assert "deployed_waiting_for_v21" in source
