from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_write_json, canonical_json_bytes, sha256_bytes, utc_now
from .kaggle import FixtureRunner, KaggleCommandError, KaggleRunner
from .provenance import unknown_notebook


SCHEMA_VERSION = 1


def _decimal_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower().removesuffix("h")
    try:
        return format(Decimal(text), "f")
    except InvalidOperation:
        return None


def _numeric_score(value: Any) -> str | None:
    try:
        return format(Decimal(str(value)), "f")
    except (InvalidOperation, ValueError):
        return None


def _records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in ("items", "results", "submissions", "leaderboard", "topics", "kernels"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, Mapping)]
    return []


def normalize_quota(raw: Any) -> dict[str, Any]:
    resources: dict[str, dict[str, Any]] = {}
    for row in _records(raw):
        name = str(row.get("resource", "unknown")).upper()
        resources[name] = {
            "used_hours": _decimal_text(row.get("used")),
            "remaining_hours": _decimal_text(row.get("remaining")),
            "total_hours": _decimal_text(row.get("total")),
            "refresh_at": row.get("refreshAt"),
        }
    return {"resources": resources}


def normalize_submissions(raw: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _records(raw):
        rows.append(
            {
                "ref": str(item.get("ref", "")),
                "file_name": str(item.get("fileName", "")),
                "submitted_at": item.get("date"),
                "description": str(item.get("description", "")),
                "status": str(item.get("status", "")),
                "public_score": _numeric_score(item.get("publicScore")),
                "private_score": _numeric_score(item.get("privateScore")),
            }
        )
    return rows


def normalize_leaderboard(raw: Any) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(_records(raw), start=1):
        score = item.get("score", item.get("publicScore"))
        team = item.get("teamName", item.get("team", item.get("name", "")))
        rank = item.get("rank", index)
        rows.append(
            {
                "rank": int(rank) if str(rank).isdigit() else None,
                "team": str(team),
                "score": _numeric_score(score),
                "last_submission_at": item.get("lastSubmissionDate", item.get("date")),
                "ref": str(item.get("teamId", team)),
            }
        )
    # Stable sorts encode score desc, timestamp desc, reference asc without
    # lossy timestamp arithmetic or reversing the reference tie-breaker.
    rows.sort(key=lambda row: row["ref"])
    rows.sort(key=lambda row: str(row.get("last_submission_at") or ""), reverse=True)
    rows.sort(
        key=lambda row: Decimal(row["score"]) if row["score"] is not None else Decimal("-Infinity"),
        reverse=True,
    )
    return rows


def normalize_topics(raw: Any) -> list[dict[str, Any]]:
    rows = [
        {
            "id": str(item.get("id", "")),
            "title": str(item.get("title", "")),
            "author": str(item.get("authorName", item.get("author", ""))),
            "comments": int(item.get("commentCount", 0) or 0),
            "votes": int(item.get("votes", 0) or 0),
            "posted_at": item.get("postDate"),
        }
        for item in _records(raw)
    ]
    rows.sort(key=lambda row: (str(row["posted_at"] or ""), row["id"]), reverse=True)
    return rows


def normalize_notebooks(raw: Any) -> list[dict[str, Any]]:
    return [
        {
            **unknown_notebook(str(item.get("ref", "")), str(item.get("title", ""))),
            "author": str(item.get("author", "")),
            "last_run_at": item.get("lastRunTime"),
            "votes": int(item.get("totalVotes", 0) or 0),
            "cli_order": index,
        }
        for index, item in enumerate(_records(raw))
    ]


def _command_specs(slug: str, notebook_limit: int) -> dict[str, list[str]]:
    return {
        "quota": ["quota", "--format", "json"],
        "submissions": [
            "competitions", "submissions", slug, "--format", "json", "--page-size", "200"
        ],
        "leaderboard": [
            "competitions", "leaderboard", slug, "--show", "--format", "json", "--page-size", "200"
        ],
        "topics": [
            "competitions", "topics", "list", slug, "--format", "json", "--sort-by", "recent"
        ],
        "kernels": [
            "kernels", "list", "--competition", slug, "--format", "json",
            "--sort-by", "scoreDescending", "--page-size", str(notebook_limit)
        ],
    }


def collect_snapshot(
    config: Mapping[str, Any],
    *,
    fixture_dir: Path | None = None,
    live: bool = False,
    notebook_limit: int = 20,
    now: datetime | None = None,
) -> dict[str, Any]:
    if live == (fixture_dir is not None):
        raise ValueError("select exactly one data source: --live or --fixture-dir")
    runner = KaggleRunner() if live else FixtureRunner(Path(fixture_dir))
    raw: dict[str, Any] = {}
    statuses: dict[str, dict[str, str]] = {}
    for name, arguments in _command_specs(str(config["slug"]), notebook_limit).items():
        try:
            raw[name] = runner.run_json(arguments)
            statuses[name] = {"status": "ok"}
        except (KaggleCommandError, OSError, ValueError, KeyError) as exc:
            raw[name] = []
            statuses[name] = {"status": "unavailable", "reason": str(exc)[:300]}

    collected = now or utc_now()
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "competition_slug": config["slug"],
        "collected_at": collected.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_mode": "live" if live else "fixture",
        "collection_status": statuses,
        "quota": normalize_quota(raw["quota"]),
        "submissions": normalize_submissions(raw["submissions"]),
        "leaderboard": normalize_leaderboard(raw["leaderboard"]),
        "topics": normalize_topics(raw["topics"]),
        "notebooks": normalize_notebooks(raw["kernels"]),
    }
    snapshot["content_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
    return snapshot


def persist_snapshot(snapshot: Mapping[str, Any], root: Path) -> Path:
    timestamp = str(snapshot["collected_at"]).replace(":", "").replace("-", "").replace(".", "")
    stem = f"{timestamp}-{snapshot['content_sha256'][:12]}"
    target = root / ".biohub" / "snapshots" / f"{stem}.json"
    suffix = 1
    while target.exists():
        target = root / ".biohub" / "snapshots" / f"{stem}-{suffix}.json"
        suffix += 1
    return atomic_write_json(target, dict(snapshot))


def status_line(snapshot: Mapping[str, Any]) -> str:
    gpu = snapshot.get("quota", {}).get("resources", {}).get("GPU", {})
    scores = [
        Decimal(row["public_score"])
        for row in snapshot.get("submissions", [])
        if row.get("public_score") is not None
    ]
    best = format(max(scores), "f") if scores else "unavailable"
    remaining = gpu.get("remaining_hours") or "unavailable"
    return (
        f"GPU remaining: {remaining}h | best public score: {best} | "
        f"notebooks reviewed: {len(snapshot.get('notebooks', []))}"
    )
