from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "scripts/build-focus3d-bridge-proposals-v1.py"))


def test_proposal_notebook_is_label_blind_and_fail_closed() -> None:
    notebook = MODULE["build_notebook"]()
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "focus3d_bridge_proposals_terminal.json" in source
    assert "ground_truth_opened=False" in source
    assert "require_tracks=True" not in source
    assert "per_sample_metrics(" not in source
    assert "submission.csv" in source
    assert "unexpectedly created submission.csv" in source
    assert notebook["metadata"]["codex"]["submission_command_included"] is False
