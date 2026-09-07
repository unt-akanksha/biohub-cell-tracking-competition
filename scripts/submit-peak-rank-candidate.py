#!/usr/bin/env python
"""Submit one externally promoted temporal peak-ranking candidate."""

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
RUN_ID = "peak-rank-tracking-candidate-v1"
DAILY_SUBMISSION_LIMIT = 5


def validate_promotion(path: Path, *, expected_run_id: str = RUN_ID):
    promotion = json.loads(path.read_text(encoding="utf-8"))
    submission = Path(str(promotion.get("submission_path", ""))).resolve()
    if not (
        promotion.get("schema_version") == 1
        and promotion.get("status") == "eligible_for_submission"
        and promotion.get("run_id") == expected_run_id
        and float(promotion.get("target_public_score", 0.0)) == 0.945
        and promotion.get("known_public_hash_match") is False
        and promotion.get("worker_count") == 2
        and float(promotion.get("proxy_gain", 0.0)) >= 0.003
        and float(promotion.get("adjusted_edge_delta", -1.0)) >= -0.001
        and float(promotion.get("worst_movie_proxy_delta", -1.0)) >= -0.005
        and isinstance(promotion.get("runtime_manifest_sha256"), str)
        and len(promotion["runtime_manifest_sha256"]) == 64
        and isinstance(promotion.get("checkpoint_sha256"), str)
        and len(promotion["checkpoint_sha256"]) == 64
        and promotion.get("competition_submission_performed") is False
        and promotion.get("authorized_for_submission") is True
        and submission.is_file()
        and sha256_file(submission) == promotion.get("submission_sha256")
    ):
        raise RuntimeError("peak-ranking promotion evidence is invalid")
    return promotion, submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    parser.add_argument("--expected-run-id", default=RUN_ID)
    parser.add_argument(
        "--message",
        default=(
            "Independent temporal peak-ranking detector plus attributed harmonic "
            "Trackastra; clean complete-movie promotion"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    promotion_path = args.promotion.resolve()
    promotion, submission = validate_promotion(
        promotion_path, expected_run_id=args.expected_run_id
    )
    if args.kernel_version < 1:
        raise ValueError("kernel version must be positive")
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise FileExistsError(receipt)
    eligible = {
        "schema_version": 1,
        "status": "eligible",
        "run_id": args.expected_run_id,
        "competition": COMPETITION,
        "kernel_ref": args.kernel_ref,
        "kernel_version": args.kernel_version,
        "submission_path": str(submission),
        "submission_sha256": promotion["submission_sha256"],
        "promotion_path": str(promotion_path),
        "promotion_sha256": sha256_file(promotion_path),
        "runtime_manifest_sha256": promotion["runtime_manifest_sha256"],
        "checkpoint_sha256": promotion["checkpoint_sha256"],
        "proxy_gain": promotion["proxy_gain"],
        "competition_submission_performed": False,
    }
    if not args.execute:
        print(json.dumps(eligible, indent=2, sort_keys=True))
        return
    daily_count = current_daily_submission_count()
    if daily_count >= DAILY_SUBMISSION_LIMIT:
        raise RuntimeError(f"daily submission limit reached: {daily_count}/{DAILY_SUBMISSION_LIMIT}")
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
            "kaggle", "competitions", "submit", "-c", COMPETITION,
            "-f", str(submission), "-m", args.message,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        atomic_json(
            receipt,
            {
                **intent,
                "status": "submission_command_failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "kaggle_cli_stdout": completed.stdout.strip(),
                "kaggle_cli_stderr": completed.stderr.strip(),
            },
        )
        raise RuntimeError("Kaggle submission command failed; inspect receipt")
    atomic_json(
        receipt,
        {
            **intent,
            "status": "submitted",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "competition_submission_performed": True,
            "kaggle_cli_stdout": completed.stdout.strip(),
            "kaggle_cli_stderr": completed.stderr.strip(),
        },
    )
    print(receipt)


if __name__ == "__main__":
    main()
