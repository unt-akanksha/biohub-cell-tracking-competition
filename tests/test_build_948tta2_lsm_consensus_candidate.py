from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-948tta2-lsm-consensus-candidate.py"


def test_candidate_is_hash_pinned_topology_preserving_and_metric_hack_free() -> None:
    values = runpy.run_path(str(SCRIPT))
    notebook = values["build_notebook"]()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert values["sha256_file"](values["SOURCE_NOTEBOOK"]) == values["SOURCE_NOTEBOOK_SHA256"]
    assert "redoctopusk/biohub-948tta2" in source
    assert "exact_equal_final_integer_coordinate" in source
    assert "lsm_consensus_node_count_changes" in source
    assert "candidate/control validator coverage is incomplete" in source.lower()
    assert '"metric_hack_used": False' in source
    assert '"estimated_node_count_used_by_candidate": False' in source
    assert "negative-time" not in source.lower()
    assert "-10000" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"candidate-cell-{index}", "exec")


def test_candidate_metadata_is_private_offline_two_gpu_shape() -> None:
    values = runpy.run_path(str(SCRIPT))
    values["main"]()
    metadata = json.loads(
        (values["TARGET_DIR"] / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == [
        values["FEATURE24_KERNEL"],
        values["FEATURE36_KERNEL"],
    ]
    assert values["RUNTIME_REF"] in metadata["dataset_sources"]
