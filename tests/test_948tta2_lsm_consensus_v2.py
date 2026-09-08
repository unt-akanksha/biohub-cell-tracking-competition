from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-948tta2-lsm-consensus-candidate-v2.py"


def test_v2_repairs_bounds_provenance_without_weakening_integrity() -> None:
    values = runpy.run_path(str(BUILDER))
    notebook = values["build_notebook"]()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "lsm_consensus_preexisting_out_of_bounds" in source
    assert "lsm_consensus_remaining_out_of_bounds" in source
    assert "(~after_valid & before_valid).sum()" in source
    assert "proposal24 = np.clip(proposal24, 0, maximum)" in source
    assert "proposal36 = np.clip(proposal36, 0, maximum)" in source
    assert "lsm_consensus_node_count_changes" in source
    assert "lsm_consensus_edge_changes" in source
    assert '"metric_hack_used": False' in source
    assert "948tta2-lsm-consensus-v1" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"candidate-v2-cell-{index}", "exec")


def test_v2_metadata_and_controller_are_fail_closed() -> None:
    values = runpy.run_path(str(BUILDER))
    values["main"]()
    metadata = json.loads(
        (values["TARGET_DIR"] / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["id"] == "indarkarhana/biohub-948tta2-lsm-consensus-v2"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"

    controller = ROOT / "scripts/wait-verify-submit-948tta2-lsm-consensus-v2.ps1"
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(controller),
            "-ValidateOnly",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status":  "valid"' in completed.stdout


def test_v2_verifier_and_submitter_share_run_id() -> None:
    verifier = runpy.run_path(
        str(ROOT / "scripts/verify-948tta2-lsm-consensus-candidate-v2.py")
    )
    submitter = runpy.run_path(
        str(ROOT / "scripts/submit-948tta2-lsm-consensus-candidate-v2.py")
    )
    assert verifier["RUN_ID"] == submitter["RUN_ID"]
    assert verifier["RUN_ID"] == "948tta2-lsm-consensus-v2"
