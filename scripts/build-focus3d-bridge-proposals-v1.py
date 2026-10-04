"""Build label-blind FOCUS proposal generation for the frozen bridge split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / "scripts/build-focus3d-frozen-validation-v1.py"))
POLICY_PATH = ROOT / "research/focus3d_bridge_rescue_v1_policy.json"
TARGET_ID = "biohub-focus3d-bridge-proposals-v1"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"


TERMINAL = r'''# Bind proposal graphs without opening labels or creating a submission.
import hashlib as _bp_hashlib

def _bp_tree_hash(root):
    digest = _bp_hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()

_bp_paths = [Path(predict_dir) / f"{stem}.geff" for stem in valid_id]
if len(_bp_paths) != 4 or any(not path.exists() for path in _bp_paths):
    raise RuntimeError(f"Frozen bridge proposals are incomplete: {_bp_paths}")
if Path("/kaggle/working/submission.csv").exists():
    raise RuntimeError("Proposal-only run unexpectedly created submission.csv")
_FV_FINISHED = True
_fv_timer.cancel()
_fv_write(
    "completed",
    proposal_graph_sha256={path.stem: _bp_tree_hash(path) for path in _bp_paths},
    complete_movie_count=len(_bp_paths),
    ground_truth_opened=False,
    focus3d_source_notebook_sha256=
        "459cedbe2068fff6b4dfc57e20a9103a14bbc059baa4612c9f7693433a618de4",
    focus3d_code_license="BSD-3-Clause",
    focus3d_weights_license="Apache-2.0",
)
print("Frozen FOCUS bridge proposals complete and hash-bound.")
'''


def build_notebook() -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    stems = tuple(policy["validation_stems"])
    globals_map = BASE["build_notebook"].__globals__
    old_stems = globals_map["FROZEN_STEMS"]
    globals_map["FROZEN_STEMS"] = stems
    try:
        notebook = BASE["build_notebook"]()
    finally:
        globals_map["FROZEN_STEMS"] = old_stems
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "_FV_RUN_ID = \"focus3d-frozen-validation-v1\"" in source:
            source = source.replace(
                'focus3d-frozen-validation-v1', 'focus3d-bridge-proposals-v1'
            ).replace(
                'focus3d_frozen_validation_terminal.json',
                'focus3d_bridge_proposals_terminal.json',
            )
            source = source.replace(
                '[\n            "44b6_d754aa59", "44b6_7a302da0",\n'
                '            "6bba_debd7bfa", "6bba_fc5f39dc",\n        ]',
                repr(list(stems)),
            )
            cell["source"] = source.splitlines(keepends=True)
        elif "# Patched official complete-movie evaluation" in source:
            cell["source"] = TERMINAL.splitlines(keepends=True)
    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    if "require_tracks=True" in joined or "per_sample_metrics(" in joined:
        raise RuntimeError("Proposal notebook retained ground-truth scoring")
    notebook["metadata"]["codex"] = {
        "run_id": "focus3d-bridge-proposals-v1",
        "frozen_policy_sha256": BASE["sha256_file"](POLICY_PATH),
        "frozen_stems": list(stems),
        "ground_truth_opened": False,
        "submission_command_included": False,
    }
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if TARGET_DIR.exists():
        if not args.replace:
            raise FileExistsError(TARGET_DIR)
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = build_notebook()
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")), encoding="ascii"
    )
    source_metadata = json.loads(BASE["SOURCE_METADATA"].read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub FOCUS3D Bridge Proposals v1",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "machine_shape": "NvidiaTeslaT4",
    }
    metadata.pop("id_no", None)
    (TARGET_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()
