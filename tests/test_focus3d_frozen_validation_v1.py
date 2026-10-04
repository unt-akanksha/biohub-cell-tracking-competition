from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-focus3d-frozen-validation-v1.py"


def test_focus3d_validation_is_frozen_patched_and_evaluation_only() -> None:
    values = runpy.run_path(str(SCRIPT), run_name="focus3d_validation_test")
    notebook = values["build_notebook"]()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert values["sha256_file"](values["SOURCE_NOTEBOOK"]) == values[
        "SOURCE_NOTEBOOK_SHA256"
    ]
    assert values["sha256_file"](values["SOURCE_METADATA"]) == values[
        "SOURCE_METADATA_SHA256"
    ]
    for stem in values["FROZEN_STEMS"]:
        assert stem in source
    assert "44b6_0113de3b" not in source
    assert 'MODE = "local"' in source
    assert "FOCUS3D_COMPILE = False" in source
    assert values["PATCHED_METRICS_SHA256"] in source
    assert values["PATCHED_DIVISION_METRICS_SHA256"] in source
    assert "summarise(metric_rows)" in source
    assert '"metric_hack_used": False' in source
    assert '"competition_test_data_read": False' in source
    assert '"authorized_for_submission": False' in source
    assert 'SUBMISSION_PATH = "submission.csv"' not in source
    assert "writer.writerow" not in source
    assert "kaggle competitions submit" not in source.lower()
    assert notebook["metadata"]["codex"]["submission_command_included"] is False
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"focus3d-{index}", "exec")
