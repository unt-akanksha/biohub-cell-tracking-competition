from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-public-node-refinement.py"
RUN_ID = "lsm-fm-public-node-refinement-v2"
RUNTIME_NAME = "biohub-lsm-fm-public-node-runtime-v2"


def main() -> None:
    values = runpy.run_path(str(BASE_BUILDER))
    base_main = values["main"]
    target = ROOT / "kaggle" / f"biohub-{RUN_ID}"
    base_main.__globals__.update(
        {
            "RUN_ID": RUN_ID,
            "KAGGLE_SLUG": "biohub-lsm-fm-public-node-preset-v2",
            "KAGGLE_TITLE": "Biohub LSM-FM Public Node Preset v2",
            "RUNTIME_NAME": RUNTIME_NAME,
            "RUNTIME": ROOT / ".biohub" / "staging" / RUNTIME_NAME,
            "TARGET": target,
            "NOTEBOOK": target / f"biohub-{RUN_ID}.ipynb",
            "PREDECLARED_STRATEGY": "public_lsm_r2_p2_b025",
        }
    )
    base_main()


if __name__ == "__main__":
    main()
