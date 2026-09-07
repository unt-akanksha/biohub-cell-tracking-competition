from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARVEST = (
    ROOT
    / "scripts/wait-harvest-antelume-peak-rank-temporal-min-local-snr-balanced-v23.ps1"
)
VERIFY = (
    ROOT
    / "scripts/verify-antelume-peak-rank-temporal-min-local-snr-balanced-v23-harvest.py"
)


def test_harvest_is_hash_verified_and_submission_free() -> None:
    source = HARVEST.read_text(encoding="utf-8")
    assert "sha256sum -c" in source
    assert "harvest.verified" in source
    assert "accepted_for_kaggle_validation" in source
    assert "Start-Process -FilePath ssh.exe" in source
    assert "-WindowStyle Hidden" in source
    assert "kaggle competitions submit" not in source


def test_verifier_freezes_temporal_minimum_local_snr_contract() -> None:
    source = VERIFY.read_text(encoding="utf-8")
    assert "83_812_614" in source
    assert "5_000" in source
    assert "12_568_331" in source
    assert "temporal_min_local_snr_safe_rank_multiscale_global_peak_rank_v23" in source
    assert '"temporal_reducer": "minimum"' in source
    assert '"local_snr_bands": [[3, 9], [5, 13], [7, 15]]' in source
    assert "duplicate_44b6_once_balance_embryo_crops" in source
