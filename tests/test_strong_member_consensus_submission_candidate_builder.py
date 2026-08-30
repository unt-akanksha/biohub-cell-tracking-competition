from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(
    str(ROOT / "scripts/build-ranked-consensus-submission-candidate.py")
)
SCRIPT = ROOT / "scripts/build-strong-member-consensus-submission-candidate.py"
MODULE = runpy.run_path(str(SCRIPT))


def test_v2_templates_compile_and_load_dynamic_members() -> None:
    compile(
        MODULE["MODEL_SETUP_TEMPLATE"].replace("__MANIFEST_SHA256__", "a" * 64),
        "<strong-member-setup>",
        "exec",
    )
    compile(MODULE["RANKED_HELPERS"], "<strong-member-ranking>", "exec")
    compile(MODULE["CANDIDATE_EVIDENCE"], "<strong-member-evidence>", "exec")

    assert "_RCD_DEEP_MODELS = []" in MODULE["MODEL_SETUP_TEMPLATE"]
    assert "deep_rank_matrix.mean(axis=0)" in MODULE["RANKED_HELPERS"]
    assert '"base_safe_division_heuristic_enabled"' in MODULE["CANDIDATE_EVIDENCE"]
    assert '"safe_division_heuristic_disabled"' not in MODULE["CANDIDATE_EVIDENCE"]


def test_shared_transform_keeps_clean_base_divisions_enabled(tmp_path: Path) -> None:
    fake_builder = tmp_path / "fake_dataset_builder.py"
    fake_builder.write_text(
        "def verify_dataset(root):\n"
        "    return {'manifest_sha256': 'a' * 64}\n",
        encoding="utf-8",
    )
    notebook = COMMON["transform_notebook"](
        tmp_path,
        dataset_builder_path=fake_builder,
        watchdog_run_id=MODULE["RUN_ID"],
        keep_base_safe_divisions=True,
        attribution=MODULE["ATTRIBUTION"],
        model_setup_template=MODULE["MODEL_SETUP_TEMPLATE"],
        ranked_helpers=MODULE["RANKED_HELPERS"],
        candidate_evidence=MODULE["CANDIDATE_EVIDENCE"],
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "1"' in code
    assert '"BIOHUB_OUTPUT_SAFE_DIVISIONS": "1"' in code
    assert 'os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "0"' not in code
    assert "_RCD_DEEP_MODELS = []" in code
    assert MODULE["RUN_ID"] in code
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), "<candidate-cell>", "exec")


def test_v2_metadata_requires_offline_two_t4_shape() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"enable_gpu": True' in source
    assert '"enable_tpu": False' in source
    assert '"enable_internet": False' in source
    assert '"machine_shape": "NvidiaTeslaT4"' in source
    assert "Exactly two T4 GPUs are required" in MODULE["MODEL_SETUP_TEMPLATE"]
