from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify-real-localization-expanded-replay-archive.py"
HARVESTER = ROOT / "scripts/wait-harvest-kaggle-real-localization-expanded-replay-v2.ps1"


def test_expanded_replay_verifier_is_streaming_and_fail_closed() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    assert "tarfile.open(archive_path, mode=\"r\")" in source
    assert "archive.extractfile(member)" in source
    assert "archive.extractall" not in source
    assert "member.issym" not in source
    assert 'ROLE_COUNTS = {"optimization": 480, "selection": 17, "sealed_audit": 14}' in source
    assert "a80c9028b21fdba1746bc2686dc5c64f0852e7954d1e758ba1357bcca5aa0edc" in source


def test_expanded_replay_harvester_is_versioned_and_never_submits() -> None:
    source = HARVESTER.read_text(encoding="utf-8")
    assert "ExpectedKernelVersion = 2" in source
    assert '"$KernelRef/$ExpectedKernelVersion"' in source
    assert "--file-pattern" in source
    assert "verify-real-localization-expanded-replay-archive.py" in source
    assert '"harvest-v$ExpectedKernelVersion-terminal.json"' in source
    assert "kaggle competitions submit" not in source
    assert "competition_submission_performed = $false" in source
