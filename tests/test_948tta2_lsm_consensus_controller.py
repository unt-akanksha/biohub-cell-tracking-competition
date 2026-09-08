from __future__ import annotations

from pathlib import Path
import runpy
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_controller_is_one_shot_and_promotion_gated() -> None:
    script = ROOT / "scripts/wait-verify-submit-948tta2-lsm-consensus.ps1"
    text = script.read_text(encoding="utf-8")
    assert "candidate_rejected" in text
    assert "eligible_for_submission" in text
    assert "--execute" in text
    assert "PollSeconds must be at least 60" in text
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-ValidateOnly",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status":  "valid"' in completed.stdout


def test_verifier_and_submitter_are_fail_closed() -> None:
    verifier = runpy.run_path(
        str(ROOT / "scripts/verify-948tta2-lsm-consensus-candidate.py")
    )
    submitter = runpy.run_path(
        str(ROOT / "scripts/submit-948tta2-lsm-consensus-candidate.py")
    )
    assert verifier["RUN_ID"] == submitter["RUN_ID"]
    assert verifier["FEATURE24_MODEL_SHA256"] != verifier["FEATURE36_MODEL_SHA256"]
    source = (ROOT / "scripts/verify-948tta2-lsm-consensus-candidate.py").read_text(
        encoding="utf-8"
    )
    assert "Nonconsecutive edge" in source
    assert "metric_hack_used" in source
    assert "min(deltas.values()) >= 0.0" in source
