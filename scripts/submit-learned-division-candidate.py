#!/usr/bin/env python
"""Submit one externally verified learned-division candidate exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


COMPETITION = "biohub-cell-tracking-during-development"
RUN_ID = "ema-learned-division-candidate-v1"
DAILY_SUBMISSION_LIMIT = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def submission_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def current_daily_submission_count() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:  # pragma: no cover - operator environment
        raise RuntimeError("Kaggle Python SDK is unavailable") from error
    api = KaggleApi()
    api.authenticate()
    today = datetime.now(timezone.utc).date()
    count = 0
    for row in api.competition_submissions(COMPETITION):
        observed = submission_date(getattr(row, "date", None))
        if observed is not None and observed.astimezone(timezone.utc).date() == today:
            count += 1
    return count


def validate_promotion(path: Path) -> tuple[dict[str, Any], Path]:
    promotion = json.loads(path.read_text(encoding="utf-8"))
    submission = Path(str(promotion.get("submission_path", ""))).resolve()
    if not (
        promotion.get("schema_version") == 1
        and promotion.get("status") == "eligible_for_submission"
        and promotion.get("run_id") == RUN_ID
        and float(promotion.get("target_public_score", 0.0)) == 0.945
        and promotion.get("known_public_hash_match") is False
        and int(promotion.get("learned_edges_added", 0)) > 0
        and int(promotion.get("learned_reassignments", -1)) == 0
        and int(promotion.get("learned_node_or_coordinate_changes", -1)) == 0
        and float(promotion.get("proxy_gain", 0.0)) >= 0.005
        and promotion.get("competition_submission_performed") is False
        and promotion.get("authorized_for_submission") is True
        and submission.is_file()
        and sha256_file(submission) == promotion.get("submission_sha256")
    ):
        raise RuntimeError("learned-division promotion evidence is invalid")
    return promotion, submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    parser.add_argument(
        "--message",
        default="0.945 candidate: EMA plus external learned division recovery",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    promotion_path = args.promotion.resolve()
    promotion, submission = validate_promotion(promotion_path)
    if args.kernel_version < 1:
        raise ValueError("kernel version must be positive")
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise FileExistsError(f"submission receipt already exists: {receipt}")
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
        "learned_edges_added": promotion["learned_edges_added"],
        "competition_submission_performed": False,
    }
    if not args.execute:
        print(json.dumps(eligible, indent=2, sort_keys=True))
        return
    daily_count = current_daily_submission_count()
    if daily_count >= DAILY_SUBMISSION_LIMIT:
        raise RuntimeError(
            f"daily submission limit reached: {daily_count}/{DAILY_SUBMISSION_LIMIT}"
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
