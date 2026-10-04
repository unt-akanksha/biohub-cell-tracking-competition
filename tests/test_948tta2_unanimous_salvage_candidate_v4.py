from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = runpy.run_path(
    str(ROOT / "scripts/build-948tta2-unanimous-salvage-candidate-v4.py")
)
RUNTIME = ROOT / ".biohub/cache/graph-context-unanimous-salvage-v4-deploy"


def test_builds_fail_closed_cross_family_evaluation() -> None:
    notebook = SCRIPT["build_notebook"](RUNTIME)
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "GRAPH_CONTEXT_SALVAGE_MANIFEST.json" in source
    assert "BIOHUB_UNANIMOUS_SALVAGE_ENABLE" in source
    assert "all_member_thresholds_passed" in source
    assert "model.predict_proba(features)" in source
    assert "row[\"model\"].predict_proba(features)" not in source
    assert "graph-context-unanimous-salvage-policy.json" in source
    assert "graph-context-fresh-policy.json" not in source
    assert '_GCD_POLICY["morphology_model_sha256"]' not in source
    assert "kaggle competitions submit" not in source
    assert '"competition_submission_performed": False' in source
    assert list(SCRIPT["FROZEN_VALIDATION_STEMS"]) == [
        "44b6_12dfb391", "44b6_267148e4", "6bba_062c8d37", "6bba_07e24132"
    ]
    assert notebook["metadata"]["codex"]["validation_stems_seen_by_graph_training"] is False


def test_metadata_requests_private_two_t4_offline_kernel(tmp_path: Path) -> None:
    metadata = json.loads(
        (ROOT / ".biohub/research/public-refresh-20260908-r1/redoctopusk__biohub-948tta2/kernel-metadata.json").read_text()
    )
    assert metadata["enable_internet"] is False
    assert SCRIPT["RUNTIME_REF"].startswith("indarkarhana/")
