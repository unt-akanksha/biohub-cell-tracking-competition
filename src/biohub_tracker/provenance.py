from __future__ import annotations

from enum import StrEnum


class ProvenanceClass(StrEnum):
    EXPLICIT_METRIC_HACK = "explicit_metric_hack"
    REPRODUCED_POST_PATCH = "reproduced_post_patch"
    REPORTED_POST_PATCH = "reported_post_patch"
    STALE_OR_GHOST_RISK = "stale_or_ghost_risk"
    AUTOMATED_NO_KNOWN_SIGNATURE = "automated_no_known_signature"
    UNKNOWN = "unknown"


def unknown_notebook(ref: str, title: str) -> dict[str, str]:
    return {
        "ref": ref,
        "title": title,
        "provenance": ProvenanceClass.UNKNOWN.value,
        "evidence": "source_not_audited",
    }

