from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".biohub" / "staging" / "biohub-lsm-fm-public-node-runtime-v1"
KERNEL = ROOT / "kaggle" / "biohub-lsm-fm-public-node-refinement-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_public_node_runtime_is_hash_bound() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build-lsm-fm-public-node-runtime.py"), "--replace"],
        check=True,
        cwd=ROOT,
    )
    manifest = json.loads((RUNTIME / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    evaluator = RUNTIME / "evaluate_public_node_refinement.py"
    assert evaluator.is_file()
    assert manifest["files"][evaluator.name]["sha256"] == sha256_file(evaluator)
    assert manifest["candidate"]["node_ids_preserved"] is True
    assert manifest["candidate"]["edges_preserved"] is True


def test_public_node_kernel_has_no_submission_path() -> None:
    if not RUNTIME.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build-lsm-fm-public-node-runtime.py")],
            check=True,
            cwd=ROOT,
        )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build-lsm-fm-public-node-refinement.py")],
        check=True,
        cwd=ROOT,
    )
    notebook_path = KERNEL / "biohub-lsm-fm-public-node-refinement-v1.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="ascii"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="ascii"))
    assert "evaluate_public_node_refinement.py" in source
    assert "--public-predictions" in source
    assert '"public_graph_used_as_base": True' in source
    assert '"exact_public_replica": False' in source
    assert "kaggle competitions submit" not in source.lower()
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"][0] == "indarkarhana/biohub-lsm-fm-public-node-runtime-v1"
