from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build-lsm-fm-pu-adaptation-v2.py"
RUN_ID = "lsm-fm-image-text-pu-adaptation-v1"
TARGET = ROOT / "kaggle" / f"biohub-{RUN_ID}"
NOTEBOOK = TARGET / f"biohub-{RUN_ID}.ipynb"
OLD_CHECKPOINT_SHA256 = "d287049e5f86ad1db7350cdf30acf2309c9dca34be8570c398f330f346afcfb0"
CHECKPOINT_SHA256 = "aca3c5d43ef7f3d7ed2ff169d1ab72b71a03acec293a48283d73d38fcf3520e7"
RUNTIME_MANIFEST_SHA256 = "5b73f47d4695492ebad504539ee1be9538faa259115a848f1ecd348c74154721"


def replace_all(notebook: dict, old: str, new: str, *, minimum: int = 1) -> int:
    matches = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] not in {"code", "markdown"}:
            continue
        source = "".join(cell["source"])
        observed = source.count(old)
        if observed:
            source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)
            matches += observed
    if matches < minimum:
        raise RuntimeError(
            f"base builder drift for {old!r}: expected at least {minimum}, observed {matches}"
        )
    return matches


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
    replace_all(
        notebook,
        "biohub-lsm-fm-pu-runtime-v1",
        "biohub-lsm-fm-image-text-pu-runtime-v1",
        minimum=2,
    )
    replace_all(
        notebook,
        "lsm_fm_image_only_student.pt",
        "lsm_fm_image_text_student.pt",
    )
    replace_all(notebook, OLD_CHECKPOINT_SHA256, CHECKPOINT_SHA256, minimum=3)
    replace_all(
        notebook,
        "fe6077dd0618f4fd2e6b026d88bf18323a66b27b7cebd5bb9431b2e1d3d6735c",
        RUNTIME_MANIFEST_SHA256,
    )
    replace_all(notebook, "lsm_fm_pu_model", "lsm_fm_image_text_pu_model")
    replace_all(notebook, "lsm_fm_pu_validation", "lsm_fm_image_text_pu_validation")
    replace_all(notebook, '"--steps", "768"', '"--steps", "3072"')
    replace_all(notebook, '"--min-steps", "192"', '"--min-steps", "768"')
    replace_all(
        notebook,
        '"--encoder-unfreeze-step", "192"',
        '"--encoder-unfreeze-step", "768"',
    )
    replace_all(notebook, '"--encoder-blocks", "2"', '"--encoder-blocks", "4"')
    replace_all(notebook, '"--pairs-per-movie", "1"', '"--pairs-per-movie", "2"')
    replace_all(
        notebook,
        '"--decoder-learning-rate", "2e-4"',
        '"--decoder-learning-rate", "1e-4"',
    )
    replace_all(
        notebook,
        '"--encoder-learning-rate", "2e-6"',
        '"--encoder-learning-rate", "1e-6"',
    )
    replace_all(
        notebook,
        "The openly licensed LSM-FM image-only student initializes a new one-channel Biohub heatmap head",
        "The openly licensed feature-36 LSM-FM image-text student initializes a new one-channel Biohub heatmap head",
    )
    replace_all(
        notebook,
        "Launching independent LSM-FM PU adaptation:",
        "Launching independent multimodal LSM-FM PU adaptation:",
    )
    replace_all(
        notebook,
        "LSM-FM PU training and clean validation complete; no submission was created.",
        "Multimodal LSM-FM PU training and clean validation complete; no submission was created.",
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )

    metadata_path = TARGET / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    metadata["title"] = "Biohub LSM-FM Image-Text PU Adaptation v1"
    metadata["dataset_sources"] = [
        (
            "indarkarhana/biohub-lsm-fm-image-text-pu-runtime-v1"
            if value == "indarkarhana/biohub-lsm-fm-pu-runtime-v1"
            else value
        )
        for value in metadata["dataset_sources"]
    ]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
