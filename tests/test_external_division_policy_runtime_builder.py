from __future__ import annotations

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-external-division-policy-runtime.py"


def test_packages_hash_bound_calibrator_without_submission_code(tmp_path: Path) -> None:
    module = runpy.run_path(str(BUILDER))
    output = tmp_path / "runtime"
    previous = sys.argv
    try:
        sys.argv = [str(BUILDER), "--output-root", str(output)]
        module["main"]()
    finally:
        sys.argv = previous

    verified = module["verify_runtime"](output)
    assert verified["authorized_for_kaggle_calibration"] is True
    assert verified["authorized_for_submission"] is False
    assert (output / "calibrate_division_recovery_policy.py").is_file()
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in output.glob("*.py")
    )
    assert "kaggle competitions submit" not in source
