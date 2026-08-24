from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import atomic_replace_json, atomic_replace_text
from .ledger import EventType, ExperimentEvent, RunState, RunStatus, reconstruct_runs


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _markdown(value: Any) -> str:
    text = _CONTROL.sub(" ", str(value or ""))
    return " ".join(text.split()).replace("|", "\\|").replace("<", "&lt;")


def _event_time(event: ExperimentEvent) -> datetime:
    return datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))


def _effective_payloads(state: RunState) -> dict[str, dict[str, Any]]:
    payloads = {event.event_id: dict(event.payload) for event in state.events}
    for event in state.events:
        if event.event_type is not EventType.AMENDMENT:
            continue
        target = str(event.payload.get("target_event_id"))
        replacements = event.payload.get("replacement_fields", {})
        if target in payloads and isinstance(replacements, Mapping):
            payloads[target].update(replacements)
    return payloads


def _summarize_state(state: RunState) -> dict[str, Any]:
    payloads = _effective_payloads(state)
    registration_event = next(
        event for event in state.events if event.event_type is EventType.REGISTERED
    )
    registration = payloads[registration_event.event_id]
    terminal_event = next(
        (
            event
            for event in reversed(state.events)
            if event.event_type in {EventType.COMPLETED, EventType.FAILED, EventType.REJECTED}
        ),
        None,
    )
    terminal = payloads[terminal_event.event_id] if terminal_event else {}
    decision_event = next(
        (event for event in reversed(state.events) if event.event_type is EventType.DECISION),
        None,
    )
    decision = payloads[decision_event.event_id] if decision_event else {}
    failure_reason = terminal.get("failure_reason") or terminal.get("reason")
    evidence = terminal.get("metrics") or terminal.get("evidence_summary") or {}
    next_gate = terminal.get("next_gate") or registration.get("next_gate")
    lifecycle_status = state.status.value
    visible_status = (
        "incomplete" if next_gate and decision.get("decision") != "promote" else lifecycle_status
    )
    return {
        "run_id": state.run_id,
        "registered_at": registration_event.created_at,
        "parent": registration.get("parent"),
        "hypothesis": registration.get("hypothesis", "not registered"),
        "status": visible_status,
        "lifecycle_status": lifecycle_status,
        "exact_evidence": evidence,
        "public_score_non_authoritative": terminal.get("public_score"),
        "declared_gpu_hours": registration.get("declared_max_runtime_hours"),
        "actual_gpu_hours": terminal.get("actual_runtime_hours"),
        "decision": decision.get("decision"),
        "decision_evidence": list(decision.get("evidence", [])),
        "failure_reason": failure_reason,
        "failed_gate": terminal.get("failed_gate"),
        "next_gate": next_gate,
        "dependencies": list(registration.get("dependencies", [])),
        "authorized_for_submission": bool(
            terminal.get(
                "authorized_for_submission", registration.get("authorized_for_submission", False)
            )
        ),
        "amendments": list(state.amendments),
        "imported_audit": bool(registration.get("imported_audit", False)),
    }


def select_next_gate(run_summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    summaries = list(run_summaries)
    by_id = {str(item["run_id"]): item for item in summaries}
    terminal = {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.REJECTED.value}
    candidates = [
        item
        for item in summaries
        if item.get("next_gate")
        and all(by_id.get(dep, {}).get("status") in terminal for dep in item.get("dependencies", []))
    ]
    if candidates:
        return min(candidates, key=lambda item: (str(item["registered_at"]), str(item["run_id"])))
    active = [
        item
        for item in summaries
        if item.get("lifecycle_status") in {RunStatus.REGISTERED.value, RunStatus.RUNNING.value}
    ]
    if not active:
        return None
    return max(active, key=lambda item: (str(item["registered_at"]), str(item["run_id"])))


def render_progress_json(events: Iterable[ExperimentEvent]) -> dict[str, Any]:
    runs = reconstruct_runs(events)
    summaries = [_summarize_state(runs[run_id]) for run_id in sorted(runs)]
    summaries.sort(key=lambda item: (str(item["registered_at"]), str(item["run_id"])))
    next_run = select_next_gate(summaries)
    return {
        "schema_version": 1,
        "runs": summaries,
        "next_experiment": (
            {
                "run_id": next_run["run_id"],
                "hypothesis": next_run["hypothesis"],
                "next_gate": next_run.get("next_gate") or "resume_nonterminal_run",
            }
            if next_run
            else None
        ),
    }


def _compact_evidence(value: Any) -> str:
    if not value:
        return "not recorded"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_progress_markdown(events: Iterable[ExperimentEvent]) -> str:
    projection = render_progress_json(events)
    next_experiment = projection["next_experiment"]
    lines = [
        "# Biohub Experiment Progress",
        "",
        "Public leaderboard score is non-authoritative and cannot independently promote a run.",
        "",
        "## Highest-Value Next Experiment",
        "",
    ]
    if next_experiment:
        lines.extend(
            [
                f"- Run: `{_markdown(next_experiment['run_id'])}`",
                f"- Hypothesis: {_markdown(next_experiment['hypothesis'])}",
                f"- Next gate: `{_markdown(next_experiment['next_gate'])}`",
            ]
        )
    else:
        lines.append("- No active gate is registered.")
    lines.extend(
        [
            "",
            "## Run History",
            "",
            "| Run | Parent | Hypothesis | State | Exact evidence | Public score (non-authoritative) | GPU hours actual / declared | Decision | Failure / failed gate | Next gate |",
            "|---|---|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for run in projection["runs"]:
        actual = run.get("actual_gpu_hours") or "unknown"
        declared = run.get("declared_gpu_hours") or "unknown"
        failure = run.get("failure_reason") or run.get("failed_gate") or "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_markdown(run['run_id'])}`",
                    _markdown(run.get("parent") or "—"),
                    _markdown(run["hypothesis"]),
                    _markdown(run["status"]),
                    _markdown(_compact_evidence(run["exact_evidence"])),
                    _markdown(run.get("public_score_non_authoritative") or "—"),
                    _markdown(f"{actual} / {declared}"),
                    _markdown(run.get("decision") or "—"),
                    _markdown(failure),
                    _markdown(run.get("next_gate") or "—"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Exact Evidence Details", ""])
    for run in projection["runs"]:
        lines.extend(
            [
                f"### {_markdown(run['run_id'])}",
                "",
                f"- Authorized for submission: `{str(run['authorized_for_submission']).lower()}`",
                f"- Imported audit: `{str(run['imported_audit']).lower()}`",
                f"- Evidence: `{_markdown(_compact_evidence(run['exact_evidence']))}`",
                f"- Decision evidence: `{_markdown(_compact_evidence(run['decision_evidence']))}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_progress_reports(events: Iterable[ExperimentEvent], root: Path) -> tuple[Path, Path]:
    materialized = list(events)
    markdown_path = atomic_replace_text(
        root / "reports" / "PROGRESS.md", render_progress_markdown(materialized)
    )
    json_path = atomic_replace_json(
        root / "reports" / "progress.json", render_progress_json(materialized)
    )
    return markdown_path, json_path
