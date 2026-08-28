from __future__ import annotations

import copy

from research.temporal_contrastive.verify_multiscale_pretraining_output import (
    recompute_gates,
)


def _metrics(composite: float, top1: float, mrr: float, division: float) -> dict:
    return {
        "composite": composite,
        "top1": top1,
        "mrr": mrr,
        "division_top2": division,
        "rows": 100,
        "division_rows": 10,
        "transitions": 8,
    }


def test_recomputed_multiscale_gates_require_selection_and_audit_gain() -> None:
    worker = {
        "initial_selection": _metrics(0.4, 0.4, 0.4, 0.4),
        "best_selection": _metrics(0.6, 0.6, 0.6, 0.5),
        "initial_audit": _metrics(0.3, 0.3, 0.3, 0.3),
        "final_audit": _metrics(0.5, 0.5, 0.5, 0.4),
    }
    selection, audit = recompute_gates(worker)
    assert selection["passed"] is True
    assert audit["passed"] is True

    regressed = copy.deepcopy(worker)
    regressed["final_audit"]["top1"] = 0.2
    _selection, rejected_audit = recompute_gates(regressed)
    assert rejected_audit["passed"] is False
