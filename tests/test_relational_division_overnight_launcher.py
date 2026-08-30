from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_runner_is_large_gated_and_bound_to_antelume() -> None:
    text = (ROOT / "scripts/run-antelume-relational-division-sweep-v1.sh").read_text()

    assert "--steps 15000" in text
    assert "211063,311071,411083,511091" in text
    assert text.count('--initial-model "$initial_') == 2
    assert "A10G" in text
    assert "current_probe_terminal" in text
    assert "score_relational_division_development_probe.py" in text
    assert 'if test "$status" -eq 0' in text
    assert "biohub-relational-development-inventory-v1.json" in text
    assert "while nvidia-smi" in text
    assert "__RELATIONAL_ARCHIVE_SHA256__" in text
    assert "kaggle competitions submit" not in text


def test_local_controller_waits_for_both_extraction_and_credentials() -> None:
    text = (ROOT / "scripts/wait-deploy-antelume-relational-division-sweep.ps1").read_text()

    assert "kaggle kernels status" in text
    assert "& kaggle kernels output $kernelRef" in text
    assert "reusing_existing_relational_archive" in text
    assert '"429|Too Many Requests"' in text
    assert "for ($retry = 1; -not $downloaded -and $retry -le 12; $retry++)" in text
    assert "aws sts get-caller-identity" in text
    assert "ec2-instance-connect send-ssh-public-key" in text
    assert "Start-Process" not in text
    assert "nohup bash scripts/run-antelume-relational-division-sweep-v1.sh" in text
    assert "development_probe_runs_only_after_audit_acceptance = $true" in text
    assert "competition_submission_performed = $false" in text
