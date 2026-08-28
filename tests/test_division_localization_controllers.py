from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def validate(script: str) -> str:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / script),
            "-ValidateOnly",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_localization_launch_controller_validates_without_launching() -> None:
    output = validate("wait-launch-multiscale-division-localization.ps1")

    assert '"status":  "validated"' in output
    assert '"expected_gpu_count":  2' in output
    assert '"quota_reserve_hours":  0' in output


def test_localization_verification_controller_validates_without_polling() -> None:
    output = validate("wait-verify-multiscale-division-localization.ps1")

    assert '"status":  "validated"' in output
    assert "multiscale_division_localization_verification" in output
