#!/usr/bin/env python
"""Submit one externally verified LSM-consensus candidate exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMON = runpy.run_path(str(ROOT / "scripts/submit-learned-division-candidate.py"))
atomic_json = COMMON["atomic_json"]
current_daily_submission_count = COMMON["current_daily_submission_count"]
sha256_file = COMMON["sha256_file"]
COMPETITION = "biohub-cell-tracking-during-development"
RUN_ID = "948tta2-lsm-consensus-v1"
DAILY_SUBMISSION_LIMIT = 5


def validate_promotion(path: Path):
    promotion = json.loads(path.read_text(encoding="utf-8"))
    submission = Path(str(promotion.get("submission_path", ""))).resolve()
    if not (
        promotion.get("schema_version") == 1
        and promotion.get("status") == "eligible_for_submission"
        and promotion.get("run_id") == RUN_ID
        and float(promotion.get("target_public_score", 0.0)) == 0.945
        and promotion.get("known_public_hash_match") is False
        and promotion.get("metric_hack_used") is False
        and int(promotion.get("production_coordinate_changes", 0)) > 0
        and float(promotion.get("proxy_gain", 0.0)) > 0.0
        and float(promotion.get("minimum_movie_adjusted_edge_delta", -1.0)) >= 0.0
        and promotion.get("competition_submission_performed") is False
        and promotion.get("authorized_for_submission") is True
        and submission.is_file()
        and sha256_file(submission) == promotion.get("submission_sha256")
    ):
        raise RuntimeError("LSM-consensus promotion evidence is invalid")
    return promotion, submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    parser.add_argument(
        "--message",
        default="Clean 948TTA2 + two-model exact-coordinate LSM consensus v1",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    promotion_path = args.promotion.resolve()
    promotion, submission = validate_promotion(promotion_path)
    if args.kernel_version < 1:
        raise ValueError("Kernel version must be positive")
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise FileExistsError(f"Submission receipt already exists: {receipt}")
    eligible = {
        "schema_version": 1,
        "status": "eligible",
        "run_id": RUN_ID,
        "competition": COMPETITION,
        "kernel_ref": args.kernel_ref,
        "kernel_version": args.kernel_version,
        "submission_path": str(submission),
        "submission_sha256": promotion["submission_sha256"],
        "promotion_path": str(promotion_path),
        "promotion_sha256": sha256_file(promotion_path),
        "proxy_gain": promotion["proxy_gain"],
        "production_coordinate_changes": promotion["production_coordinate_changes"],
        "competition_submission_performed": False,
    }
    if not args.execute:
        print(json.dumps(eligible, indent=2, sort_keys=True))
        return
    daily_count = current_daily_submission_count()
    if daily_count >= DAILY_SUBMISSION_LIMIT:
        raise RuntimeError(
            f"Daily submission limit reached: {daily_count}/{DAILY_SUBMISSION_LIMIT}"
        )
    intent = {
        **eligible,
        "status": "submission_intent_recorded",
        "intent_recorded_at": datetime.now(timezone.utc).isoformat(),
        "daily_submission_count_before": daily_count,
        "message": args.message,
    }
    atomic_json(receipt, intent)
    completed = subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-f",
            str(submission),
            "-m",
            args.message,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result = {
        **intent,
        "status": "submitted" if completed.returncode == 0 else "submission_command_failed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "competition_submission_performed": completed.returncode == 0,
        "kaggle_cli_stdout": completed.stdout.strip(),
        "kaggle_cli_stderr": completed.stderr.strip(),
    }
    atomic_json(receipt, result)
    if completed.returncode != 0:
        raise RuntimeError("Kaggle submission command failed; inspect receipt")
    print(receipt)


if __name__ == "__main__":
    main()
