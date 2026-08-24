from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from .io import (
    atomic_replace_json,
    atomic_replace_text,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    utc_now,
)
from .kaggle import FixtureRunner, KaggleCommandError, KaggleRunner
from .provenance import classify_notebook, load_policy, unknown_notebook


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
        score = item.get("score", item.get("Score", item.get("publicScore")))
        team = item.get(
            "teamName", item.get("TeamName", item.get("team", item.get("name", "")))
        )
        rank = item.get("rank", item.get("Rank", index))
        rows.append(
            {
                "rank": int(rank) if str(rank).isdigit() else None,
                "team": str(team),
                "score": _numeric_score(score),
                "last_submission_at": item.get(
                    "lastSubmissionDate", item.get("SubmissionDate", item.get("date"))
                ),
                "ref": str(item.get("teamId", item.get("TeamId", team))),
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
    root: Path | None = None,
    audit_notebook_sources: bool = False,
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

    raw["leaderboard_top"] = raw["leaderboard"]
    if live:
        try:
            raw["leaderboard"] = runner.download_leaderboard(str(config["slug"]))
            statuses["leaderboard_full"] = {"status": "ok"}
        except (KaggleCommandError, OSError, ValueError) as exc:
            raw["leaderboard"] = []
            statuses["leaderboard_full"] = {
                "status": "unavailable",
                "reason": str(exc)[:300],
            }
    else:
        statuses["leaderboard_full"] = {"status": statuses["leaderboard"]["status"]}

    collected = now or utc_now()
    notebooks = normalize_notebooks(raw["kernels"])
    policy_root = (root or Path.cwd()).resolve()
    registry_path = policy_root / "policies" / "notebook_audits.json"
    patterns_path = policy_root / "policies" / "metric_hack_patterns.json"
    registry = load_policy(registry_path) if registry_path.exists() else {"notebooks": []}
    patterns = load_policy(patterns_path) if patterns_path.exists() else {"patterns": []}
    source_status: dict[str, str] = {}
    for notebook in notebooks:
        source_paths: list[Path] = []
        if audit_notebook_sources:
            try:
                if live:
                    source_paths = [
                        runner.pull_kernel_source(
                            notebook["ref"], policy_root / ".biohub" / "cache" / "notebooks"
                        )
                    ]
                else:
                    owner, slug = notebook["ref"].split("/", 1)
                    candidate = Path(fixture_dir) / "notebook_sources" / owner / slug
                    if candidate.exists():
                        source_paths = [candidate]
                source_status[notebook["ref"]] = "audited" if source_paths else "unavailable"
            except (KaggleCommandError, OSError, ValueError) as exc:
                source_status[notebook["ref"]] = f"unavailable: {str(exc)[:200]}"
        audit = classify_notebook(
            notebook["ref"], notebook["title"], source_paths, registry, patterns
        )
        notebook.update(audit.to_dict())
    if audit_notebook_sources:
        statuses["notebook_sources"] = {
            "status": (
                "ok"
                if source_status and all(value == "audited" for value in source_status.values())
                else "partial"
            ),
            "reason": "; ".join(f"{key}={value}" for key, value in source_status.items()),
        }

    policy_projection = {
        "entry_deadline_utc": config.get("entry_deadline_utc"),
        "final_deadline_utc": config.get("final_deadline_utc"),
        "daily_submission_limit": config.get("daily_submission_limit"),
        "final_submission_slots": config.get("final_submission_slots"),
        "notebook_runtime_limit_hours": config.get("notebook_runtime_limit_hours"),
        "gpu_reserve_hours": config.get("gpu_reserve_hours"),
        "official_sources": config.get("official_sources", {}),
        "notebook_registry_sha256": sha256_bytes(canonical_json_bytes(registry)),
        "metric_hack_policy_sha256": sha256_bytes(canonical_json_bytes(patterns)),
    }
    policy_projection["content_sha256"] = sha256_bytes(canonical_json_bytes(policy_projection))
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "competition_slug": config["slug"],
        "collected_at": collected.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_mode": "live" if live else "fixture",
        "collection_status": statuses,
        "competition_policy": policy_projection,
        "quota": normalize_quota(raw["quota"]),
        "submissions": normalize_submissions(raw["submissions"]),
        "leaderboard": normalize_leaderboard(raw["leaderboard"]),
        "leaderboard_top": normalize_leaderboard(raw["leaderboard_top"]),
        "topics": normalize_topics(raw["topics"]),
        "notebooks": notebooks,
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


def _normalized_alias(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def build_status_projection(
    snapshot: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    collected = _parse_time(snapshot.get("collected_at"))
    collected_date = collected.date() if collected else None
    scored = [row for row in snapshot.get("submissions", []) if row.get("public_score") is not None]
    scored.sort(key=lambda row: str(row.get("ref", "")))
    scored.sort(key=lambda row: str(row.get("submitted_at") or ""), reverse=True)
    scored.sort(key=lambda row: Decimal(row["public_score"]), reverse=True)
    best_submission = scored[0] if scored else None

    submissions_today = sum(
        1
        for row in snapshot.get("submissions", [])
        if collected_date is not None
        and (submitted := _parse_time(row.get("submitted_at"))) is not None
        and submitted.date() == collected_date
    )
    daily_limit = int(config.get("daily_submission_limit", 5))

    aliases = {_normalized_alias(value) for value in config.get("team_aliases", [])}
    rank_row = next(
        (
            row
            for row in snapshot.get("leaderboard", [])
            if _normalized_alias(row.get("team")) in aliases
        ),
        None,
    )
    leaderboard_status = snapshot.get("collection_status", {}).get("leaderboard_full", {})
    if rank_row:
        rank_reason = None
    elif leaderboard_status.get("status") != "ok":
        rank_reason = leaderboard_status.get("reason", "leaderboard collection unavailable")
    else:
        rank_reason = "configured team alias is absent from the collected leaderboard page"

    gpu = snapshot.get("quota", {}).get("resources", {}).get("GPU", {})
    remaining_text = gpu.get("remaining_hours")
    reserve = Decimal(str(config.get("gpu_reserve_hours", "8.00")))
    remaining = Decimal(remaining_text) if remaining_text is not None else None
    spendable = max(Decimal("0"), remaining - reserve) if remaining is not None else None

    current = config.get("current_status", {})
    notebooks = list(snapshot.get("notebooks", []))
    return {
        "schema_version": 1,
        "generated_from_snapshot_at": snapshot.get("collected_at"),
        "snapshot_sha256": snapshot.get("content_sha256"),
        "competition": {
            "slug": snapshot.get("competition_slug"),
            "rank": rank_row.get("rank") if rank_row else None,
            "rank_unavailable_reason": rank_reason,
            "leader_score": (
                snapshot.get("leaderboard_top", [{}])[0].get("score")
                if snapshot.get("leaderboard_top")
                else None
            ),
            "best_clean_public_score": current.get("best_clean_public_score"),
            "best_clean_score_evidence": current.get(
                "best_clean_score_evidence", "clean submission provenance not registered"
            ),
            "final_deadline_utc": config.get("final_deadline_utc"),
        },
        "submissions": {
            "returned_total": len(snapshot.get("submissions", [])),
            "today": submissions_today,
            "daily_limit": daily_limit,
            "remaining_today": max(0, daily_limit - submissions_today),
            "best_public_score": best_submission.get("public_score") if best_submission else None,
            "best_submission_ref": best_submission.get("ref") if best_submission else None,
        },
        "gpu": {
            "remaining_hours": format(remaining, ".2f") if remaining is not None else None,
            "reserve_hours": format(reserve, ".2f"),
            "spendable_hours": format(spendable, ".2f") if spendable is not None else None,
            "refresh_at": gpu.get("refresh_at"),
        },
        "notebook_provenance": {
            "clean_research_candidates": [
                row for row in notebooks if row.get("disposition") == "research_candidate"
            ],
            "excluded_metric_hacks": [
                row for row in notebooks if row.get("disposition") == "excluded_metric_hack"
            ],
            "needs_source_review": [
                row for row in notebooks if row.get("disposition") == "needs_source_review"
            ],
        },
        "recent_discussions": list(snapshot.get("topics", [])),
        "active_hypothesis": current.get("active_hypothesis", "not registered"),
        "next_gate": current.get("next_gate", "not registered"),
        "competition_policy": snapshot.get("competition_policy", {}),
        "collection_status": snapshot.get("collection_status", {}),
    }


def _value_or_unavailable(value: Any, reason: str | None = None) -> str:
    if value is not None and value != "":
        return str(value)
    return f"unavailable ({reason})" if reason else "unavailable"


def render_status_report(snapshot: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    status = build_status_projection(snapshot, config)
    competition = status["competition"]
    submissions = status["submissions"]
    gpu = status["gpu"]
    provenance = status["notebook_provenance"]
    lines = [
        "# Biohub Competition Status",
        "",
        f"Generated from snapshot: `{status['generated_from_snapshot_at']}`  ",
        f"Snapshot SHA-256: `{status['snapshot_sha256']}`",
        "",
        "## Competition",
        "",
        f"- Slug: `{competition['slug']}`",
        f"- Public rank: {_value_or_unavailable(competition['rank'], competition['rank_unavailable_reason'])}",
        f"- Best clean public score: {_value_or_unavailable(competition['best_clean_public_score'])}",
        f"- Clean-score evidence: {competition['best_clean_score_evidence']}",
        f"- Current public leader score: {_value_or_unavailable(competition['leader_score'])}",
        f"- Final deadline: `{competition['final_deadline_utc']}`",
        "",
        "## GPU Safety",
        "",
        f"- Remaining: {_value_or_unavailable(gpu['remaining_hours'])} hours",
        f"- Protected reserve: {gpu['reserve_hours']} hours",
        f"- Spendable before reserve: {_value_or_unavailable(gpu['spendable_hours'])} hours",
        f"- Quota refresh: {_value_or_unavailable(gpu['refresh_at'])}",
        "",
        "## Personal Submissions",
        "",
        f"- Returned by CLI: {submissions['returned_total']}",
        f"- Submitted today (UTC): {submissions['today']} / {submissions['daily_limit']}",
        f"- Remaining daily allowance: {submissions['remaining_today']}",
        f"- Best personal public score: {_value_or_unavailable(submissions['best_public_score'])}",
        "",
        "## Notebook Provenance",
        "",
        "`research_candidate` means source was manually screened for known exploit logic; it does not reproduce the displayed score.",
        "",
        "### Clean Research Candidates",
        "",
    ]
    for row in provenance["clean_research_candidates"]:
        lines.append(f"- `{row['ref']}` — {row['provenance']}; {row['evidence']}")
    if not provenance["clean_research_candidates"]:
        lines.append("- None in the collected notebook set.")
    lines.extend(["", "### Excluded Metric Hacks", ""])
    for row in provenance["excluded_metric_hacks"]:
        lines.append(f"- `{row['ref']}` — {row['evidence']}")
    if not provenance["excluded_metric_hacks"]:
        lines.append("- None detected in the collected notebook set.")
    lines.extend(["", "### Needs Source Review", ""])
    for row in provenance["needs_source_review"]:
        lines.append(f"- `{row['ref']}` — {row['provenance']}; {row['evidence']}")
    if not provenance["needs_source_review"]:
        lines.append("- None.")
    lines.extend(["", "## Recent Discussions", ""])
    for topic in status["recent_discussions"]:
        lines.append(
            f"- {topic['posted_at']} — {topic['title']} (topic `{topic['id']}`, "
            f"{topic['comments']} comments, {topic['votes']} votes)"
        )
    if not status["recent_discussions"]:
        reason = status["collection_status"].get("topics", {}).get("reason")
        lines.append(f"- {_value_or_unavailable(None, reason)}")
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            f"- Active hypothesis: {status['active_hypothesis']}",
            f"- Gate: {status['next_gate']}",
            "",
            "## Policy Provenance",
            "",
            f"- Policy SHA-256: `{status['competition_policy'].get('content_sha256', 'unavailable')}`",
            f"- Notebook registry SHA-256: `{status['competition_policy'].get('notebook_registry_sha256', 'unavailable')}`",
            f"- Metric-hack policy SHA-256: `{status['competition_policy'].get('metric_hack_policy_sha256', 'unavailable')}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_status_reports(
    snapshot: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> tuple[Path, Path]:
    report_dir = root / "reports"
    markdown_path = atomic_replace_text(
        report_dir / "competition-status.md", render_status_report(snapshot, config)
    )
    json_path = atomic_replace_json(
        report_dir / "competition-status.json", build_status_projection(snapshot, config)
    )
    return markdown_path, json_path
