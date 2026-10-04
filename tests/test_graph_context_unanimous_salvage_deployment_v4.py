from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-graph-context-unanimous-salvage-deployment-v4.py"
DEPLOYMENT = ROOT / ".biohub/cache/graph-context-unanimous-salvage-v4-deploy"


def test_real_unanimous_salvage_deployment_verifies_evaluation_only() -> None:
    values = runpy.run_path(str(SCRIPT), run_name="salvage_deployment_test")
    result = values["verify_dataset"](DEPLOYMENT)

    assert result["status"] == "verified_evaluation_only"
    assert result["member_count"] == 4
    assert result["authorized_for_full_candidate_evaluation"] is True
    assert result["authorized_for_submission"] is False
