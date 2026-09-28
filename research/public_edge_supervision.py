"""Conservative sparse-annotation labels for ordinary candidate-edge audits.

Unmatched endpoints are unknown, never background negatives. A negative needs
both annotated endpoints and closed source-outgoing / target-incoming evidence.
This helper is for training-data diagnostics, not submission generation.
"""
from __future__ import annotations


def classify_edge(source, target, matches, truth_edges, truth_out, truth_in):
    first, second = matches.get(source), matches.get(target)
    if first is None or second is None or first == -1 or second == -1:
        return -1
    if (first, second) in truth_edges:
        return 1
    if first in truth_out and second in truth_in:
        return 0
    return -1


def mapped_pairs(edges, matches):
    return {(matches[a], matches[b]) for a, b in edges
            if matches.get(a) not in (None, -1)
            and matches.get(b) not in (None, -1)}
