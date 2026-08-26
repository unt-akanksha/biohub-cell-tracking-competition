from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-pu-adaptation.py"
RUN_ID = "lsm-fm-pu-adaptation-v2"
TARGET = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = TARGET / f"biohub-{RUN_ID}.ipynb"


def replace_notebook_source(notebook: dict, old: str, new: str) -> None:
    matches = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        observed = source.count(old)
        if observed:
            source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)
            matches += observed
    if matches != 1:
        raise RuntimeError(
            f"v1 builder drift for {old!r}: expected one match, observed {matches}"
        )


def main() -> None:
    values = runpy.run_path(str(BASE_BUILDER))
    base_main = values["main"]
    base_main.__globals__.update(
        {
            "RUN_ID": RUN_ID,
            "TARGET": TARGET,
            "NOTEBOOK": NOTEBOOK,
        }
    )
    base_main()

    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    replace_notebook_source(
        notebook,
        '[str(runtime), str(support_repo.parent), run_env.get("PYTHONPATH", "")]',
        '[str(runtime), str(monai_import_root), str(support_repo.parent), '
        'run_env.get("PYTHONPATH", "")]',
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )

    metadata_path = TARGET / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    metadata["title"] = "Biohub LSM-FM PU Adaptation v2"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
