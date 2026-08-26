from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_policy_requires_dual_gpu_and_runtime_margin() -> None:
    config = json.loads((ROOT / "config" / "competition.json").read_text())
    policy = config["submission_execution_policy"]

    assert policy["required_machine_shape"] == "NvidiaTeslaT4"
    assert policy["required_cuda_devices"] == 2
    assert policy["require_movie_sharding_across_devices"] is True
    assert float(policy["runtime_hard_stop_hours"]) < float(
        config["notebook_runtime_limit_hours"]
    )
    assert policy["require_non_replica_output"] is True
