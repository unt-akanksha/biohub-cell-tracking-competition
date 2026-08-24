from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TEXT_SUFFIXES = {".py", ".ipynb", ".md", ".txt", ".json", ".yaml", ".yml"}


class ProvenanceClass(StrEnum):
    EXPLICIT_METRIC_HACK = "explicit_metric_hack"
    REPRODUCED_POST_PATCH = "reproduced_post_patch"
    REPORTED_POST_PATCH = "reported_post_patch"
    STALE_OR_GHOST_RISK = "stale_or_ghost_risk"
    AUTOMATED_NO_KNOWN_SIGNATURE = "automated_no_known_signature"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceAudit:
    source_sha256: str | None
    files: tuple[str, ...]
    matched_patterns: tuple[str, ...]


@dataclass(frozen=True)
class NotebookAudit:
    ref: str
    title: str
    provenance: str
    evidence: str
    disposition: str
    source_sha256: str | None = None
    matched_patterns: tuple[str, ...] = ()
    score_reproduced: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["matched_patterns"] = list(self.matched_patterns)
        return value


def load_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"policy must be a JSON object: {path}")
    return value


def _source_files(paths: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.add(path.resolve())
        elif path.is_dir():
            files.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES
            )
    return sorted(files, key=lambda item: item.as_posix())


def audit_source_tree(
    source_paths: Iterable[str | Path], patterns: Mapping[str, Any]
) -> SourceAudit:
    files = _source_files(source_paths)
    if not files:
        return SourceAudit(None, (), ())
    compiled = []
    for entry in patterns.get("patterns", []):
        if entry.get("scope") not in {"source", "both"}:
            continue
        compiled.append(
            (str(entry["id"]), re.compile(str(entry["regex"]), re.IGNORECASE | re.MULTILINE))
        )
    digest = hashlib.sha256()
    matched: set[str] = set()
    names: list[str] = []
    for path in files:
        payload = path.read_bytes()
        stable_name = path.as_posix()
        names.append(stable_name)
        digest.update(stable_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        text = payload.decode("utf-8", errors="replace")
        for pattern_id, pattern in compiled:
            if pattern.search(text):
                matched.add(pattern_id)
    return SourceAudit(digest.hexdigest(), tuple(names), tuple(sorted(matched)))


def _registry_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(entry["ref"]): entry
        for entry in registry.get("notebooks", [])
        if isinstance(entry, Mapping) and entry.get("ref")
    }


def _title_matches(title: str, patterns: Mapping[str, Any]) -> tuple[str, ...]:
    matches = []
    for entry in patterns.get("patterns", []):
        if entry.get("scope") not in {"title", "both"}:
            continue
        if re.search(str(entry["regex"]), title, re.IGNORECASE | re.MULTILINE):
            matches.append(str(entry["id"]))
    return tuple(sorted(matches))


def classify_notebook(
    ref: str,
    title: str,
    source_paths: Sequence[str | Path] | None,
    registry: Mapping[str, Any],
    patterns: Mapping[str, Any] | None = None,
) -> NotebookAudit:
    pattern_policy = patterns or {"patterns": []}
    source_audit = audit_source_tree(source_paths or (), pattern_policy)
    matches = tuple(sorted(set(source_audit.matched_patterns + _title_matches(title, pattern_policy))))
    curated = _registry_index(registry).get(ref)

    if matches or (curated and curated.get("classification") == ProvenanceClass.EXPLICIT_METRIC_HACK):
        evidence = "matched explicit metric-hack policy" if matches else str(curated.get("evidence"))
        return NotebookAudit(
            ref,
            title,
            ProvenanceClass.EXPLICIT_METRIC_HACK.value,
            evidence,
            "excluded_metric_hack",
            source_audit.source_sha256,
            matches,
            False,
        )

    if curated:
        expected_hash = curated.get("reviewed_sha256")
        if expected_hash and source_audit.source_sha256 != expected_hash:
            reason = (
                "unknown_source_changed"
                if source_audit.source_sha256
                else "source_not_audited_for_curated_hash"
            )
            return NotebookAudit(
                ref,
                title,
                ProvenanceClass.UNKNOWN.value,
                reason,
                "needs_source_review",
                source_audit.source_sha256,
                matches,
                False,
            )
        provenance = str(curated.get("classification", ProvenanceClass.UNKNOWN.value))
        if provenance in {
            ProvenanceClass.REPRODUCED_POST_PATCH.value,
            ProvenanceClass.REPORTED_POST_PATCH.value,
            ProvenanceClass.STALE_OR_GHOST_RISK.value,
        }:
            return NotebookAudit(
                ref,
                title,
                provenance,
                str(curated.get("evidence", "curated review")),
                str(curated.get("disposition", "needs_source_review")),
                source_audit.source_sha256,
                matches,
                bool(curated.get("score_reproduced", False)),
            )

    if source_audit.source_sha256:
        return NotebookAudit(
            ref,
            title,
            ProvenanceClass.AUTOMATED_NO_KNOWN_SIGNATURE.value,
            "static_scan_no_known_signature_not_a_clean_score_claim",
            "needs_source_review",
            source_audit.source_sha256,
            matches,
            False,
        )
    return NotebookAudit(
        ref,
        title,
        ProvenanceClass.UNKNOWN.value,
        "source_not_audited",
        "needs_source_review",
        None,
        matches,
        False,
    )


def unknown_notebook(ref: str, title: str) -> dict[str, Any]:
    return classify_notebook(ref, title, None, {"notebooks": []}).to_dict()
