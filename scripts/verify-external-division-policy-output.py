#!/usr/bin/env python
"""Verify a Kaggle external-policy output and atomically publish its policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "external-division-recovery-policy-kaggle-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_file(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}, saw {matches}")
    return matches[0]


def verify_output(root: Path) -> dict[str, Any]:
    policy_path = unique_file(root, "policy.json")
    terminal_path = unique_file(root, "policy-kernel-terminal.json")
    watchdog_path = unique_file(root, "policy-watchdog-terminal.json")
    policy_module = runpy.run_path(
        str(ROOT / "scripts/build-learned-division-recovery-runtime.py")
    )
    policy = policy_module["validate_policy"](policy_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    watchdog = json.loads(watchdog_path.read_text(encoding="utf-8"))
    policy_sha256 = sha256_file(policy_path)
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == RUN_ID
        and terminal.get("host_gpu_count") == 2
        and terminal.get("calibration_visible_gpu_count") == 1
        and terminal.get("policy_sha256") == policy_sha256
        and terminal.get("frozen_division_logit_threshold")
        == policy["frozen_division_logit_threshold"]
        and terminal.get("selection") == policy["selection"]
        and terminal.get("audit") == policy["audit"]
        and terminal.get("competition_data_read") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise RuntimeError("external policy kernel terminal is invalid")
    if not (
        watchdog.get("schema_version") == 1
        and watchdog.get("status") == "completed"
        and watchdog.get("run_id") == RUN_ID
        and watchdog.get("policy_exists") is True
        and watchdog.get("policy_sha256") == policy_sha256
        and float(watchdog.get("elapsed_seconds", 7000.0)) < 6900.0
        and watchdog.get("competition_data_read") is False
        and watchdog.get("submission_created") is False
    ):
        raise RuntimeError("external policy watchdog terminal is invalid")
    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": policy_sha256,
        "policy_status": policy["status"],
        "frozen_division_logit_threshold": policy[
            "frozen_division_logit_threshold"
        ],
        "selection": policy["selection"],
        "audit": policy["audit"],
        "policy_kernel_terminal_sha256": sha256_file(terminal_path),
        "watchdog_terminal_sha256": sha256_file(watchdog_path),
        "competition_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_candidate_handoff": True,
        "authorized_for_submission": False,
    }


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(
                f"A different external policy already exists: {destination}"
            )
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--publish-policy", type=Path, required=True)
    args = parser.parse_args()
    result = verify_output(args.output_root)
    policy_path = Path(result["policy_path"])
    atomic_copy(policy_path, args.publish_policy)
    result["published_policy_path"] = str(args.publish_policy.resolve())
    result["published_policy_sha256"] = sha256_file(args.publish_policy)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
