from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile-temporal-multiscale-training-memory.py"
ARTIFACT = (
    ROOT
    / "artifacts"
    / "profiles"
    / "temporal-multiscale-training-memory-v1.json"
)


def test_training_memory_profile_passes_both_t4_cases() -> None:
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert result["status"] == "profiled"
    assert result["passed"] is True
    assert result["decision"] == "retain_published_v4_training_configuration"
    assert result["external_inventory"]["shards"] == 80
    assert result["external_inventory"]["max_nodes"] == 160
    assert result["external_inventory"]["max_candidate_edges"] == 3_551
    assert result["transfer_caps"] == {
        "max_sources": 48,
        "max_targets": 128,
        "max_nodes": 176,
        "dense_candidate_edges": 6_144,
    }
    assert result["measurements"]["v4"]["parameter_count"] == 46_386_607
    assert all(
        case["within_conservative_t4_budget"] is True
        for case in result["cases"].values()
    )
    assert result["gpu_used"] is False
    assert result["public_leaderboard_used_for_selection"] is False
    assert result["competition_submission_performed"] is False


def test_saved_tensor_profiler_excludes_parameter_storage() -> None:
    namespace = runpy.run_path(str(SCRIPT))
    import torch

    model = torch.nn.Linear(4, 2)

    def operation() -> torch.Tensor:
        return model(torch.ones(3, 4)).square().mean()

    result = namespace["saved_tensor_inventory"](
        operation, excluded_tensors=tuple(model.parameters())
    )
    assert result["logical_bytes"] > 0
    assert result["unique_storage_bytes"] > 0
    assert result["saved_tensor_count"] > 0


def test_training_memory_profiler_contains_no_external_action() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    assert "kaggle competitions" not in source
    assert "competitions submit" not in source
    assert "kaggle kernels" not in source
    assert "torch.cuda" not in source
