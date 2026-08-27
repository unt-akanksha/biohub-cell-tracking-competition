from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-temporal-patch-dual-fold-kernel.py"
KERNEL_DIR = ROOT / "kaggle" / "biohub-temporal-patch-dual-fold-v1"
NOTEBOOK = KERNEL_DIR / "biohub-temporal-patch-dual-fold-v1.ipynb"
METADATA = KERNEL_DIR / "kernel-metadata.json"


def test_temporal_patch_kernel_is_strict_two_gpu_training_only() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    metadata = json.loads(METADATA.read_text(encoding="ascii"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    compile(code, str(NOTEBOOK), "exec")
    assert metadata["id"] == "indarkarhana/biohub-temporal-patch-dual-fold-v1"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "indarkarhana/biohub-temporal-patch-runtime-v1",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
    ]
    assert metadata["kernel_sources"] == [
        "josefreitasalvesneto/biohub-synthetic-dataset",
        "indarkarhana/biohub-trackastra-dual-fold-synthetic-v1",
    ]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "torch.cuda.device_count() != 2" in code
    assert '"--steps", "30000"' in code
    assert '"--max-wall-seconds", "36000"' in code
    assert '"--orchestrator-hard-stop-seconds", "37800"' in code
    assert "DECLARED_BUDGET_SECONDS = 39_600" in code
    assert "--allow-pretrained-control" in code
    assert "Synthetic coverage is incomplete" in code
    assert "Real-movie coverage is incomplete" in code
    assert 'expected_real_inventory = {"44b6": 71, "6bba": 128}' in code
    assert "shutil.copytree(source, runtime / source.name)" in code
    assert "Ambiguous runtime directory and archive" in code
    assert "Dense 176-patch production forward/backward" in code
    assert "probe_loss.backward()" in code
    assert "weights_only=True" in code
    assert "19_221_954" in code
    assert "public_leaderboard_used_for_selection" in code
    assert "submission_created" in code
    assert "competitions submit" not in code
    assert "kaggle competitions" not in code


def test_temporal_patch_kernel_builder_is_byte_deterministic() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    first_notebook = NOTEBOOK.read_bytes()
    first_metadata = METADATA.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], check=True)

    assert NOTEBOOK.read_bytes() == first_notebook
    assert METADATA.read_bytes() == first_metadata


def test_temporal_patch_runtime_materializer_accepts_kaggle_directory_and_zip_mounts(
    tmp_path: Path,
) -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="ascii"))
    setup = "".join(notebook["cells"][2]["source"])
    tree = ast.parse(setup)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "materialize_runtime_input"
    )
    namespace = {"shutil": shutil, "zipfile": zipfile}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    materialize = namespace["materialize_runtime_input"]

    directory_input = tmp_path / "directory_input"
    nested = directory_input / "trackastra_source" / "trackastra"
    nested.mkdir(parents=True)
    (directory_input / "verify_runtime.py").write_text("pass\n", encoding="utf-8")
    (directory_input / "dataset-metadata.json").write_text("{}\n", encoding="utf-8")
    (nested / "__init__.py").write_text("\n", encoding="utf-8")
    directory_output = tmp_path / "directory_output"
    materialize(directory_input, directory_output)
    assert (directory_output / "trackastra_source" / "trackastra" / "__init__.py").is_file()
    assert (directory_output / "verify_runtime.py").is_file()
    assert not (directory_output / "dataset-metadata.json").exists()

    zip_input = tmp_path / "zip_input"
    zip_input.mkdir()
    with zipfile.ZipFile(zip_input / "trackastra_source.zip", "w") as archive:
        archive.writestr("trackastra/__init__.py", "\n")
    zip_output = tmp_path / "zip_output"
    materialize(zip_input, zip_output)
    assert (zip_output / "trackastra_source" / "trackastra" / "__init__.py").is_file()
