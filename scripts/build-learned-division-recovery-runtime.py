#!/usr/bin/env python
"""Build the private runtime for learned division recovery inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "learned-division-recovery-runtime-v1"
DATASET_ID = "indarkarhana/biohub-learned-division-recovery-runtime-v1"
SOURCE_NAMES = (
    "patch_model.py",
    "pair_fusion.py",
    "transition_context.py",
    "contextual_pair_fusion.py",
    "multiscale_contextual_pair_fusion.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(policy.get("frozen_division_logit_threshold", math.nan))
    models = policy.get("model_sha256", {})
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "accepted"
        and policy.get("run_id") == "external-division-recovery-policy-v1"
        and policy.get("appearance_family")
        == "temporal_multiscale_contextual_pair_fusion_v4"
        and math.isfinite(threshold)
        and set(models) == {"target_44b6", "target_6bba"}
        and all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in models.values()
        )
        and policy.get("audit_opened_after_threshold_freeze") is True
        and policy.get("competition_data_read") is False
        and policy.get("public_leaderboard_used_for_selection") is False
        and policy.get("submission_created") is False
        and policy.get("authorized_for_competition_graph_evaluation") is True
        and policy.get("authorized_for_submission") is False
    ):
        raise ValueError("external learned division policy is ineligible")
    return policy


def verify_runtime(root: Path) -> dict[str, Any]:
    manifest_path = root / "RUNTIME_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy_path = root / "division-recovery-policy.json"
    policy = validate_policy(policy_path)
    files = manifest.get("files", {})
    expected_names = {*SOURCE_NAMES, "learned_division_recovery.py"}
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("run_id") == RUN_ID
        and manifest.get("policy_run_id") == policy["run_id"]
        and manifest.get("policy_sha256") == sha256_file(policy_path)
        and manifest.get("competition_data_read") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and set(files) == expected_names
    ):
        raise ValueError("learned division runtime manifest is invalid")
    for name in sorted(expected_names):
        path = root / name
        if not path.is_file() or sha256_file(path) != files[name]["sha256"]:
            raise ValueError(f"learned division runtime source changed: {name}")
    return {
        "run_id": RUN_ID,
        "files": len(files),
        "manifest_sha256": sha256_file(manifest_path),
        "policy_sha256": sha256_file(policy_path),
        "frozen_division_logit_threshold": float(
            policy["frozen_division_logit_threshold"]
        ),
        "authorized_for_kernel_staging": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".biohub/staging/biohub-learned-division-recovery-runtime-v1",
    )
    args = parser.parse_args()
    policy = validate_policy(args.policy)
    args.output_root.mkdir(parents=True, exist_ok=False)
    temporal_root = ROOT / "research/temporal_contrastive"
    sources = {
        name: temporal_root / name for name in SOURCE_NAMES
    }
    sources["learned_division_recovery.py"] = (
        ROOT / "research/learned_division_recovery.py"
    )
    for name, source in sources.items():
        shutil.copy2(source, args.output_root / name)
    shutil.copy2(args.policy, args.output_root / "division-recovery-policy.json")
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "policy_run_id": policy["run_id"],
        "policy_sha256": sha256_file(
            args.output_root / "division-recovery-policy.json"
        ),
        "competition_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_command_included": False,
        "files": {
            name: {
                "path": name,
                "sha256": sha256_file(args.output_root / name),
            }
            for name in sorted(sources)
        },
    }
    write_json(args.output_root / "RUNTIME_MANIFEST.json", manifest)
    write_json(
        args.output_root / "dataset-metadata.json",
        {
            "title": "Biohub Learned Division Recovery Runtime v1",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    print(json.dumps(verify_runtime(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
