from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-948tta2-fresh-graph-candidate-v3.py"


def test_candidate_is_frozen_paired_clean_and_compilable(
    tmp_path: Path,
) -> None:
    values = runpy.run_path(str(SCRIPT), run_name="fresh_graph_candidate_test")
    manifest_sha256 = "a" * 64
    values["DEPLOY"]["verify_dataset"] = lambda root: {
        "manifest_sha256": manifest_sha256,
        "authorized_for_full_candidate_evaluation": True,
    }

    notebook = values["build_notebook"](tmp_path)
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert values["sha256_file"](values["SOURCE_NOTEBOOK"]) == values[
        "SOURCE_NOTEBOOK_SHA256"
    ]
    assert manifest_sha256 in source
    assert "redoctopusk/biohub-948tta2" in source
    assert "_stem_has_gt_division" not in source
    for stem in values["FROZEN_VALIDATION_STEMS"]:
        assert stem in source
    assert '"control"' in source
    assert '"fresh_graph"' in source
    assert '"metric_hack_used": False' in source
    assert '"public_predictions_copied": False' in source
    assert '"exact_public_replica": False' in source
    assert '"leaderboard_used_for_candidate_selection": False' in source
    assert "minimum_movie_adjusted_edge_delta" in source
    assert "production_edges_added" in source
    assert "ranked_consensus_reassignment_performed" in source
    assert "ranked_consensus_node_or_coordinate_changes" in source
    assert "kaggle competitions submit" not in source.lower()
    assert notebook["metadata"]["codex"]["submission_command_included"] is False
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"candidate-{index}", "exec")


def test_candidate_rejects_unapproved_graph_runtime(tmp_path: Path) -> None:
    values = runpy.run_path(str(SCRIPT), run_name="fresh_graph_candidate_reject_test")
    values["DEPLOY"]["verify_dataset"] = lambda root: {
        "manifest_sha256": "b" * 64,
        "authorized_for_full_candidate_evaluation": False,
    }

    try:
        values["build_notebook"](tmp_path)
    except RuntimeError as exc:
        assert "not authorized" in str(exc)
    else:
        raise AssertionError("Unapproved fresh graph runtime was accepted")
