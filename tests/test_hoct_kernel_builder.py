from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle" / "biohub-hoct-multibackbone-probe-v1"
EXPECTED_NODES = {
    "44b6_12dfb391": 44139,
    "44b6_267148e4": 21768,
    "6bba_062c8d37": 5812,
    "6bba_07e24132": 26204,
}


def test_multibackbone_notebook_has_verified_topology_cache_and_fallback() -> None:
    notebook = json.loads(
        (KERNEL / "biohub-hoct-multibackbone-probe-v1.ipynb").read_text(
            encoding="ascii"
        )
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    compile(code, str(KERNEL / "biohub-hoct-multibackbone-probe-v1.ipynb"), "exec")
    assert "validated_cached_topology" in code
    assert "EXPECTED_DEEPCENTER_SHA256" in code
    assert 'report.get("processed_validation_sha256") == csv_sha256' in code
    assert 'report.get("ground_truth_read_for_postprocessing") is False' in code
    assert "No valid cached topology found; using the exact in-kernel materializer." in code
    assert 'subprocess.run(materialize_command, check=True)' in code


def test_multibackbone_kernel_attaches_only_expected_preceding_kernel() -> None:
    metadata = json.loads(
        (KERNEL / "kernel-metadata.json").read_text(encoding="ascii")
    )

    assert metadata["kernel_sources"] == [
        "indarkarhana/biohub-trackastra-raw-confidence-acceptance-v2"
    ]
    assert metadata["enable_gpu"] is True
    assert metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is False


def _cache_validator() -> tuple[dict, object]:
    notebook = json.loads(
        (KERNEL / "biohub-hoct-multibackbone-probe-v1.ipynb").read_text(
            encoding="ascii"
        )
    )
    training_cell = "".join(notebook["cells"][3]["source"])
    function_source = training_cell.split("cached_processed_csv =", 1)[0]
    graph_manifest = {
        "files": {
            "public_preset_source.py": {"sha256": "preset-hash"},
            "public_config_source.py": {"sha256": "config-hash"},
            "public_postprocess_source.py": {"sha256": "postprocess-hash"},
        }
    }
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "graph_manifest": graph_manifest,
    }
    exec(function_source, namespace)
    return graph_manifest, namespace["validated_cached_topology"]


def _write_valid_cache(root: Path) -> Path:
    root.mkdir()
    (root / "launcher_terminal.json").write_text(
        json.dumps(
            {
                "run_id": "trackastra-raw-confidence-acceptance-v2",
                "status": "failed",
                "submission_created": False,
            }
        ),
        encoding="utf-8",
    )
    processed = root / "processed_validation"
    processed.mkdir()
    csv_path = processed / "processed_validation.csv"
    csv_path.write_text("id,dataset\n0,44b6_12dfb391\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "completed",
        "source_notebook": "evgendvorkin/biohub-0-927-lb",
        "role": "frozen comparator postprocessing only",
        "public_preset_source_sha256": "preset-hash",
        "public_config_source_sha256": "config-hash",
        "public_postprocess_source_sha256": "postprocess-hash",
        "processed_validation_sha256": hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest(),
        "deepcenter_checkpoint": {
            "sha256": "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0",
            "expected_epoch": 2,
        },
        "datasets": {
            stem: {
                "processed_nodes": count,
                "expected_processed_nodes": count,
                "processed_node_tolerance": max(1, math.ceil(count * 0.005)),
            }
            for stem, count in EXPECTED_NODES.items()
        },
        "ground_truth_read_for_postprocessing": False,
        "public_leaderboard_used_for_selection": False,
    }
    (processed / "processed_validation_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return csv_path


def test_cached_topology_accepts_hash_bound_artifact_after_later_model_failure(
    tmp_path: Path,
) -> None:
    _graph_manifest, validator = _cache_validator()
    csv_path = _write_valid_cache(tmp_path / "cache")

    assert validator(tmp_path / "cache") == csv_path


def test_cached_topology_tamper_or_malformed_report_falls_back(
    tmp_path: Path,
) -> None:
    _graph_manifest, validator = _cache_validator()
    csv_path = _write_valid_cache(tmp_path / "cache")
    csv_path.write_text("tampered", encoding="utf-8")
    assert validator(tmp_path / "cache") is None

    report_path = csv_path.with_name("processed_validation_report.json")
    report_path.write_text("{malformed", encoding="utf-8")
    assert validator(tmp_path / "cache") is None


def test_cached_topology_allows_only_bounded_node_count_drift(tmp_path: Path) -> None:
    _graph_manifest, validator = _cache_validator()
    csv_path = _write_valid_cache(tmp_path / "cache")
    report_path = csv_path.with_name("processed_validation_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["datasets"]["44b6_267148e4"]["processed_nodes"] = 21_843
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert validator(tmp_path / "cache") == csv_path

    report["datasets"]["44b6_267148e4"]["processed_nodes"] = 21_878
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert validator(tmp_path / "cache") is None
