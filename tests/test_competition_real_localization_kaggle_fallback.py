from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "research/export_competition_real_localization_labels.py"
BUILDER = ROOT / "scripts/build-real-localization-replay-cache-kernel.py"
VALIDATOR = ROOT / "scripts/validate-real-localization-shard-cache.py"
CONTROLLER = ROOT / "scripts/wait-harvest-kaggle-real-localization-replay-cache.ps1"
DATASET_REF = "indarkarhana/biohub-real-localization-labels-v2"


def test_label_exporter_is_train_only_and_hash_bound() -> None:
    source = EXPORTER.read_text(encoding="utf-8")

    assert "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1" in source
    assert '"competition_test_data_read": False' in source
    assert '"public_leaderboard_used_for_selection": False' in source
    assert '"authorized_for_submission": False' in source
    assert "FINAL_PROBE_STEMS" in source
    assert "graph_plain" in source


def test_kernel_builder_emits_private_cpu_train_only_notebook(tmp_path: Path) -> None:
    label_root = tmp_path / "labels"
    label_root.mkdir()
    (label_root / "upload_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset_id": DATASET_REF,
                "inventory_sha256": "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1",
                "archive": {
                    "path": "biohub_real_localization_labels_v1.tar.gz",
                    "sha256": "a" * 64,
                },
                "labels_manifest_sha256": "b" * 64,
                "support_wheel": {
                    "path": "numcodecs-0.16.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                    "sha256": "c" * 64,
                },
                "competition_test_data_read": False,
                "authorized_for_submission": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "kernel"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--label-upload-root",
            str(label_root),
            "--output-root",
            str(output),
        ],
        check=True,
    )

    metadata = json.loads((output / "kernel-metadata.json").read_text())
    notebook = json.loads((output / metadata["code_file"]).read_text())
    source = "".join(notebook["cells"][0]["source"])

    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [DATASET_REF]
    assert metadata["competition_sources"] == [
        "biohub-cell-tracking-during-development"
    ]
    assert "INPUT_ROOT.rglob" not in source
    assert 'train_root.name != "train"' in source
    assert '"competition_test_data_read": False' in source
    assert '"public_leaderboard_used_for_selection": False' in source
    assert '"submission_created": False' in source
    assert '"authorized_for_submission": False' in source
    assert "submission.csv" not in source
    assert '"pip", "install", "--no-index", "--no-deps"' in source
    assert "from numcodecs import Blosc" in source
    assert "import zarr" not in source
    assert "competition Zarr contract" in source
    compile(source, metadata["code_file"], "exec")


def test_shard_validator_accepts_only_complete_hash_bound_cache(tmp_path: Path) -> None:
    roles = [("optimization", 146), ("selection", 17), ("sealed_audit", 14)]
    rows = []
    for role, count in roles:
        for index in range(count):
            relative = f"{role}/movie_{role}_{index:03d}.npz"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            rows.append(
                {
                    "path": relative,
                    "role": role,
                    "stem": f"movie_{role}_{index:03d}",
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    by_role = {
        role: {
            "shards": count,
            "center_nodes": count,
            "division_critical_center_nodes": count,
        }
        for role, count in roles
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": "competition-real-localization-shards-v1",
        "source_mode": "kaggle_cpu_direct_competition_train",
        "inventory_sha256": "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1",
        "excluded_final_probe_stems": [
            "44b6_12dfb391",
            "44b6_267148e4",
            "6bba_062c8d37",
            "6bba_07e24132",
        ],
        "files": rows,
        "summary": {
            "shards": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "by_role": by_role,
        },
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    (tmp_path / "real_localization_shard_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(completed.stdout)
    assert evidence["status"] == "complete"
    assert evidence["shards"] == 177
    assert evidence["source_mode"] == "kaggle_cpu_direct_competition_train"


def test_harvest_controller_is_cpu_only_non_submission_and_fail_closed() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "kaggle kernels status" in source
    assert "kaggle kernels output" in source
    assert "wait-launch-harvest-aws-4gpu-temporal-localizer-v1.ps1" in source
    assert "wait-build-launch-aws-temporal-localizer-v2\\.ps1" in source
    assert 'kaggle_gpu_used = $false' in source
    assert 'competition_test_data_read = $false' in source
    assert 'competition_submission_performed = $false' in source
    assert "competitions submit" not in source
    assert "Stop-ExactOvernightOrchestrator" in source
    assert "quarantined partial canonical cache" in source
