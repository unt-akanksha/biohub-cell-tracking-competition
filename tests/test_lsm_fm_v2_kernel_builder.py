from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-lsm-fm-pu-adaptation-v2.py"


def test_v2_repairs_only_subprocess_monai_path(tmp_path: Path) -> None:
    values = runpy.run_path(str(BUILDER))
    target = tmp_path / "kernel"
    notebook = target / "notebook.ipynb"
    values["main"].__globals__.update({"TARGET": target, "NOTEBOOK": notebook})
    values["main"]()

    built = json.loads(notebook.read_text(encoding="ascii"))
    metadata = json.loads((target / "kernel-metadata.json").read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in built["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(notebook), "exec")
    assert 'RUN_ID = "lsm-fm-pu-adaptation-v2"' in code
    assert (
        '[str(runtime), str(monai_import_root), str(support_repo.parent), '
        'run_env.get("PYTHONPATH", "")]'
    ) in code
    assert code.count("run_env[\"PYTHONPATH\"]") == 1
    assert '"--model-family", "lsm_fm"' in code
    assert '"--batch-size", "1"' in code
    assert "competitions submit" not in code
    assert metadata["id"] == "indarkarhana/biohub-lsm-fm-pu-adaptation-v2"
    assert metadata["title"] == "Biohub LSM-FM PU Adaptation v2"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
