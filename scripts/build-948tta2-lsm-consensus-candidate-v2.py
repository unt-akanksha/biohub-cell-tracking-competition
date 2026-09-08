#!/usr/bin/env python
"""Build the v2 LSM-consensus candidate with provenance-aware bounds checks."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(
    str(ROOT / "scripts/build-948tta2-lsm-consensus-candidate.py")
)
TARGET_ID = "biohub-948tta2-lsm-consensus-v2"
TARGET_DIR = ROOT / "kaggle" / TARGET_ID
TARGET_NOTEBOOK = TARGET_DIR / f"{TARGET_ID}.ipynb"
RUNTIME_REF = BASE["RUNTIME_REF"]
FEATURE24_KERNEL = BASE["FEATURE24_KERNEL"]
FEATURE36_KERNEL = BASE["FEATURE36_KERNEL"]
SOURCE_NOTEBOOK = BASE["SOURCE_NOTEBOOK"]
SOURCE_METADATA = BASE["SOURCE_METADATA"]
SOURCE_NOTEBOOK_SHA256 = BASE["SOURCE_NOTEBOOK_SHA256"]
SOURCE_METADATA_SHA256 = BASE["SOURCE_METADATA_SHA256"]
sha256_file = BASE["sha256_file"]


def replace_exact(text: str, old: str, new: str, *, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"v1 candidate drifted for {old!r}: expected {count}, saw {actual}"
        )
    return text.replace(old, new, count)


def build_notebook() -> dict:
    """Apply only the bounds-provenance repair to the hash-pinned v1 candidate."""
    notebook = BASE["build_notebook"]()
    helper_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "def _apply_lsm_coordinate_consensus(" in "".join(cell.get("source", []))
    )
    helper = "".join(notebook["cells"][helper_index]["source"])
    helper = replace_exact(
        helper,
        '        "lsm_consensus_out_of_bounds": 0,\n',
        '        "lsm_consensus_out_of_bounds": 0,\n'
        '        "lsm_consensus_preexisting_out_of_bounds": 0,\n'
        '        "lsm_consensus_remaining_out_of_bounds": 0,\n',
    )
    helper = replace_exact(
        helper,
        "        base_native = np.asarray(\n"
        "            [[nodes_by_id[node_id][axis] for axis in (\"z\", \"y\", \"x\")] for node_id in node_ids],\n"
        "            dtype=np.float32,\n"
        "        )\n"
        "        base_small = base_native / scale\n",
        "        before_native = np.asarray(\n"
        "            [[nodes_by_id[node_id][axis] for axis in (\"z\", \"y\", \"x\")] for node_id in node_ids],\n"
        "            dtype=np.float64,\n"
        "        )\n"
        "        maximum = np.asarray(frame.shape, dtype=np.int64) - 1\n"
        "        before_valid = (\n"
        "            np.isfinite(before_native).all(axis=1)\n"
        "            & np.all(before_native >= 0, axis=1)\n"
        "            & np.all(before_native <= maximum, axis=1)\n"
        "        )\n"
        "        stats[\"lsm_consensus_preexisting_out_of_bounds\"] += int((~before_valid).sum())\n"
        "        base_native = before_native.astype(np.float32)\n"
        "        base_small = base_native / scale\n",
    )
    helper = replace_exact(
        helper,
        "        maximum = np.asarray(frame.shape, dtype=np.int64) - 1\n"
        "        proposal24 = np.clip(proposal24, 0, maximum)\n",
        "        proposal24 = np.clip(proposal24, 0, maximum)\n",
    )
    helper = replace_exact(
        helper,
        "        for local_index in np.flatnonzero(moved):\n"
        "            node = nodes_by_id[node_ids[int(local_index)]]\n"
        "            coordinate = proposal24[int(local_index)]\n"
        "            node[\"z\"], node[\"y\"], node[\"x\"] = map(float, coordinate)\n"
        "        del probability24, probability36, target24, target36\n",
        "        for local_index in np.flatnonzero(moved):\n"
        "            node = nodes_by_id[node_ids[int(local_index)]]\n"
        "            coordinate = proposal24[int(local_index)]\n"
        "            node[\"z\"], node[\"y\"], node[\"x\"] = map(float, coordinate)\n"
        "        after_native = np.asarray(\n"
        "            [[nodes_by_id[node_id][axis] for axis in (\"z\", \"y\", \"x\")] for node_id in node_ids],\n"
        "            dtype=np.float64,\n"
        "        )\n"
        "        after_valid = (\n"
        "            np.isfinite(after_native).all(axis=1)\n"
        "            & np.all(after_native >= 0, axis=1)\n"
        "            & np.all(after_native <= maximum, axis=1)\n"
        "        )\n"
        "        stats[\"lsm_consensus_remaining_out_of_bounds\"] += int((~after_valid).sum())\n"
        "        stats[\"lsm_consensus_out_of_bounds\"] += int((~after_valid & before_valid).sum())\n"
        "        del probability24, probability36, target24, target36\n",
    )
    helper = replace_exact(
        helper,
        "    for node in nodes_by_id.values():\n"
        "        point = np.asarray([node[\"z\"], node[\"y\"], node[\"x\"]], dtype=np.float64)\n"
        "        if not np.isfinite(point).all() or (point < 0).any():\n"
        "            stats[\"lsm_consensus_out_of_bounds\"] += 1\n",
        "",
    )
    notebook["cells"][helper_index]["source"] = helper.splitlines(keepends=True)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            "948tta2-lsm-consensus-v1", "948tta2-lsm-consensus-v2"
        )
        cell["source"] = source.splitlines(keepends=True)
    return notebook


def main() -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    notebook = build_notebook()
    TARGET_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=True, separators=(",", ":")),
        encoding="ascii",
    )
    base_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata = {
        **base_metadata,
        "id": f"indarkarhana/{TARGET_ID}",
        "title": "Biohub 948TTA2 LSM Consensus v2",
        "code_file": TARGET_NOTEBOOK.name,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "cell-tracking", "lsm-fm", "non-replica"],
        "dataset_sources": [*base_metadata["dataset_sources"], RUNTIME_REF],
        "kernel_sources": [FEATURE24_KERNEL, FEATURE36_KERNEL],
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
