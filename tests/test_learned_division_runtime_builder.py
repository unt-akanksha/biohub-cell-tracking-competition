from __future__ import annotations

import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build-learned-division-recovery-runtime.py"


def accepted_policy(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "accepted",
                "run_id": "external-division-recovery-policy-v1",
                "appearance_family": "temporal_multiscale_contextual_pair_fusion_v4",
                "model_sha256": {
                    "target_44b6": "a" * 64,
                    "target_6bba": "b" * 64,
                },
                "frozen_division_logit_threshold": 1.25,
                "audit_opened_after_threshold_freeze": True,
                "competition_data_read": False,
                "public_leaderboard_used_for_selection": False,
                "submission_created": False,
                "authorized_for_competition_graph_evaluation": True,
                "authorized_for_submission": False,
            }
        ),
        encoding="utf-8",
    )


def test_runtime_builder_packages_policy_and_exact_inference_sources(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(str(BUILDER))
    policy_path = tmp_path / "policy.json"
    accepted_policy(policy_path)
    output = tmp_path / "runtime"
    old_argv = __import__("sys").argv
    try:
        __import__("sys").argv = [
            str(BUILDER),
            "--policy",
            str(policy_path),
            "--output-root",
            str(output),
        ]
        module["main"]()
    finally:
        __import__("sys").argv = old_argv

    verified = module["verify_runtime"](output)
    metadata = json.loads((output / "dataset-metadata.json").read_text())
    assert verified["authorized_for_kernel_staging"] is True
    assert verified["authorized_for_submission"] is False
    assert verified["frozen_division_logit_threshold"] == 1.25
    assert metadata["id"] == "indarkarhana/biohub-learned-division-recovery-runtime-v1"
    assert (output / "learned_division_recovery.py").is_file()
    assert (output / "multiscale_contextual_pair_fusion.py").is_file()
