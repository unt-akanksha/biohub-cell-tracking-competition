from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-pu-adaptation.py"
CHECKPOINT_SHA256 = "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0"
MONAI_ARCHIVE_SHA256 = "207d0be1dee0ba5a164ea553bf0f49a316a08267e17c6729b595553aac46117d"


def _build(tmp_path: Path) -> tuple[dict, dict, str]:
    values = runpy.run_path(str(BUILDER))
    target = tmp_path / "kernel"
    notebook = target / "notebook.ipynb"
    values["main"].__globals__["TARGET"] = target
    values["main"].__globals__["NOTEBOOK"] = notebook
    values["main"]()
    built = json.loads(notebook.read_text(encoding="ascii"))
    metadata = json.loads((target / "kernel-metadata.json").read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in built["cells"]
        if cell["cell_type"] == "code"
    )
    return built, metadata, code


def test_lsm_fm_kernel_is_hash_bound_memory_safe_and_submission_free(tmp_path: Path) -> None:
    _notebook, _metadata, code = _build(tmp_path)

    compile(code, str(BUILDER), "exec")
    assert 'RUN_ID = "lsm-fm-pu-adaptation-v1"' in code
    assert "biohub-lsm-fm-pu-runtime-v1" in code
    assert CHECKPOINT_SHA256 in code
    assert MONAI_ARCHIVE_SHA256 in code
    assert 'sys.path.insert(0, str(monai_import_root))' in code
    assert "extracted MONAI file hash mismatch" in code
    assert '"--model-family", "lsm_fm"' in code
    assert '"--encoder-blocks", "2"' in code
    assert '"--batch-size", "1"' in code
    assert '"--output-name", "lsm_fm_pu_validation.json"' in code
    assert '"--max-wall-seconds", "5200"' in code
    assert '"--max-wall-seconds", "1300"' in code
    assert "competitions submit" not in code
    assert "Validation unexpectedly created submission artifacts" in code


def test_lsm_fm_kernel_metadata_is_private_gpu_without_network(tmp_path: Path) -> None:
    notebook, metadata, _code = _build(tmp_path)

    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == []
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "indarkarhana/biohub-lsm-fm-pu-runtime-v1" in metadata["dataset_sources"]
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is False
