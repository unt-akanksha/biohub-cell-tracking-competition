from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-zebrahub-contextual-pretrain-kernel.py"
KERNEL_DIR = ROOT / "kaggle" / "biohub-zebrahub-contextual-pretrain-v1"
NOTEBOOK = KERNEL_DIR / "biohub-zebrahub-contextual-pretrain-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"


def _notebook() -> tuple[dict, dict, str]:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, metadata, code


def test_pretraining_kernel_is_exact_two_gpu_external_training_only() -> None:
    _notebook_json, metadata, code = _notebook()
    compile(code, str(NOTEBOOK), "exec")

    assert metadata["id"] == "indarkarhana/biohub-zebrahub-contextual-pretrain-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-temporal-contextual-pair-fusion-runtime-v3",
        "indarkarhana/biohub-zebrahub-contextual-shards-v1",
    ]
    assert metadata["kernel_sources"] == []
    assert metadata["competition_sources"] == []
    assert "torch.cuda.device_count() != 2" in code
    assert "EXPECTED_DATASET_MANIFEST_SHA256" in code
    assert "d9f009518fb26a2aeebebf405485ff7cf36c7e3982cfbf9501aa98346e8b77a8" in code
    assert "EXPECTED_RUNTIME_MANIFEST_SHA256" in code
    assert '"--steps", "12000"' in code
    assert '"--max-wall-seconds", "21600"' in code
    assert '"--orchestrator-hard-stop-seconds", "22800"' in code
    assert "DECLARED_BUDGET_SECONDS = 24_000" in code
    assert '"--minimum-train-shards", "64"' in code
    assert '"--minimum-validation-shards", "16"' in code
    assert "verify_zebrahub_contextual_dataset.py" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "competitions submit" not in code
    assert "kaggle competitions" not in code
    assert "biohub-cell-tracking-during-development" not in code


def test_pretraining_kernel_builder_is_byte_deterministic() -> None:
    _notebook()
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], check=True)

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata


def _materializer(name: str):
    notebook, _metadata, _code = _notebook()
    setup = "".join(notebook["cells"][2]["source"])
    tree = ast.parse(setup)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"Path": Path, "shutil": shutil, "zipfile": zipfile}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace[name]


def test_shard_materializer_accepts_directories_and_split_archives(tmp_path: Path) -> None:
    materialize = _materializer("materialize_shard_input")
    direct = tmp_path / "direct"
    (direct / "train").mkdir(parents=True)
    (direct / "validation").mkdir()
    (direct / "DATASET_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (direct / "dataset-metadata.json").write_text("{}\n", encoding="utf-8")
    (direct / "train" / "one.npz").write_bytes(b"train")
    (direct / "validation" / "one.npz").write_bytes(b"validation")
    direct_output = tmp_path / "direct_output"
    materialize(direct, direct_output)
    assert (direct_output / "train" / "one.npz").read_bytes() == b"train"
    assert (direct_output / "validation" / "one.npz").read_bytes() == b"validation"
    assert (direct_output / "DATASET_MANIFEST.json").is_file()
    assert not (direct_output / "dataset-metadata.json").exists()

    archived = tmp_path / "archived"
    archived.mkdir()
    with zipfile.ZipFile(archived / "train.zip", "w") as handle:
        handle.writestr("one.npz", b"train")
    with zipfile.ZipFile(archived / "validation.zip", "w") as handle:
        handle.writestr("validation/one.npz", b"validation")
    archive_output = tmp_path / "archive_output"
    materialize(archived, archive_output)
    assert (archive_output / "train" / "one.npz").read_bytes() == b"train"
    assert (archive_output / "validation" / "one.npz").read_bytes() == b"validation"


def test_runtime_materializer_accepts_directory_and_zip_mounts(tmp_path: Path) -> None:
    materialize = _materializer("materialize_runtime_input")
    direct = tmp_path / "runtime_direct"
    nested = direct / "trackastra_source" / "trackastra"
    nested.mkdir(parents=True)
    (direct / "verify_runtime.py").write_text("pass\n", encoding="utf-8")
    (direct / "dataset-metadata.json").write_text("{}\n", encoding="utf-8")
    (nested / "__init__.py").write_text("\n", encoding="utf-8")
    direct_output = tmp_path / "runtime_direct_output"
    materialize(direct, direct_output)
    assert (direct_output / "trackastra_source" / "trackastra" / "__init__.py").is_file()
    assert (direct_output / "verify_runtime.py").is_file()
    assert not (direct_output / "dataset-metadata.json").exists()

    archived = tmp_path / "runtime_archived"
    archived.mkdir()
    with zipfile.ZipFile(archived / "trackastra_source.zip", "w") as handle:
        handle.writestr("trackastra/__init__.py", "\n")
    archive_output = tmp_path / "runtime_archive_output"
    materialize(archived, archive_output)
    assert (archive_output / "trackastra_source" / "trackastra" / "__init__.py").is_file()
