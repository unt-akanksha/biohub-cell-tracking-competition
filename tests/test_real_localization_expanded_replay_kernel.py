import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-real-localization-expanded-replay-cache-kernel.py"


def test_expanded_kernel_is_cpu_offline_and_optimization_only() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert "biohub-real-localization-expanded-inventory-v2" in source
    assert '"enable_gpu": False' in source
    assert '"enable_tpu": False' in source
    assert '"enable_internet": False' in source
    assert '"competition_sources": ["biohub-cell-tracking-during-development"]' in source
    assert "selection_centers_changed" in source
    assert "sealed_audit_centers_changed" in source
    assert 'INPUT_ROOT / "datasets" / "indarkarhana"' in source


def test_source_transform_binds_dynamic_expanded_counts(tmp_path: Path) -> None:
    module = runpy.run_path(str(BUILDER))
    inventory = {
        "summary": {
            "center_frames": 511,
            "required_frames": 1337,
            "by_role": {
                "optimization": {"center_frames": 480},
                "selection": {"center_frames": 17},
                "sealed_audit": {"center_frames": 14},
            },
        }
    }
    path = tmp_path / "expanded_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    source = module["expanded_kernel_source"](
        labels_manifest_sha256="a" * 64,
        support_wheel_name="numcodecs.whl",
        support_wheel_sha256="b" * 64,
        inventory_path=path,
    )
    assert "len(records) != 511" in source
    assert "len(frame_records) != 1337" in source
    assert 'counts["optimization"]["shards"] != 480' in source
    assert "competition-real-localization-expanded-shards-v2" in source
    assert "EXPECTED_PARENT_INVENTORY_SHA256" in source
    assert "biohub-real-localization-expanded-shards-v2.tar" in source
    assert 'tarfile.open(archive_path, mode="w")' in source
    assert 'terminal["archive_sha256"] = archive_sha256' in source
