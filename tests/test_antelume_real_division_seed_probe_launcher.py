from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run-antelume-real-division-seed-probe-after-ensemble-v1.sh"


def test_probe_launcher_waits_for_selection_and_never_submits() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "while ! test -f \"$ensemble_terminal\"" in source
    assert "authorized_for_development_probe" in source
    assert "skipped_after_selection_rejection" in source
    assert "score_real_division_seed_ensemble_probe.py" in source
    assert "overnight_seed_policy.py" in source
    assert "competition_test_data_read\": False" in source
    assert "submission_created\": False" in source
    assert "kaggle competitions submit" not in source
