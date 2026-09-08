#!/usr/bin/env python
"""Diagnose the fixed all-member graph-context ensemble on its audit role.

This is deliberately a CPU-only, non-promoting calculation.  The original v1
run froze a strongest-single-member policy before opening audit, so its eight
member ensemble cannot be retroactively authorized from this result.  The
diagnostic answers only whether the already precommitted fresh-seed v2 policy
is technically worth a later, sequential GPU run and fresh complete-movie
gate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive.graph_context_division_model import (
    GraphContextDivisionModel,
)
from research.temporal_contrastive.train_graph_context_division_sweep import (
    V1_RUN_ID,
    calibration_free_equal_rank_ensemble,
    eligible_metrics,
    load_role,
    passes_selection_gate,
    sha256_file,
    validate_manifest,
)


EXPECTED_MEMBER_COUNT = 8


def eligible_subset(data: tuple[Any, ...]) -> tuple[Any, ...]:
    """Keep only rows consumed by ``eligible_metrics``.

    Each graph-context candidate is scored independently from its already
    materialized context tokens, so discarding ineligible rows cannot change
    an eligible prediction or any reported metric.
    """

    indices = torch.nonzero(data[6], as_tuple=False).flatten()
    if not len(indices):
        raise RuntimeError("Graph-context role has no inference-eligible rows")
    tensors = tuple(value[indices] for value in data[:7])
    inventory = [data[7][index] for index in indices.tolist()]
    if not bool(tensors[6].all()):
        raise RuntimeError("Eligible graph-context subset changed its mask")
    return (*tensors, inventory)


@torch.inference_mode()
def predict_cpu(
    model: GraphContextDivisionModel,
    data: tuple[torch.Tensor, ...],
    *,
    batch_size: int,
) -> torch.Tensor:
    """Run the CUDA-trained model deterministically in float32 on CPU."""

    model.eval()
    patches, geometry, context, mask = data[:4]
    pieces: list[torch.Tensor] = []
    for start in range(0, len(patches), batch_size):
        stop = start + batch_size
        pieces.append(
            model(
                patches[start:stop].float(),
                geometry[start:stop].float(),
                context[start:stop].float(),
                mask[start:stop],
            ).float()
        )
    result = torch.cat(pieces).cpu()
    if result.ndim != 1 or len(result) != len(patches) or not torch.isfinite(result).all():
        raise RuntimeError("CPU graph-context prediction contract failed")
    return result


def _load_members(models_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate_path = models_root / "graph_context_division_sweep_terminal.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    members: list[dict[str, Any]] = []
    for member in sorted(aggregate.get("selection_accepted_members", [])):
        member_root = models_root / member
        terminal_path = member_root / "worker_terminal.json"
        checkpoint_path = member_root / "graph_context_model.pt"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if not (
            terminal.get("run_id") == V1_RUN_ID
            and terminal.get("member") == member
            and terminal.get("status") == "accepted_at_selection"
            and terminal.get("selection_gate_passed") is True
            and terminal.get("audit_opened") is False
            and terminal.get("public_code_copied") is False
            and terminal.get("public_predictions_copied") is False
            and terminal.get("public_leaderboard_used_for_selection") is False
            and terminal.get("model_sha256") == sha256_file(checkpoint_path)
        ):
            raise RuntimeError(f"Ineligible graph-context member: {member}")
        members.append(
            {
                "member": member,
                "model_sha256": terminal["model_sha256"],
                "checkpoint": checkpoint_path,
            }
        )
    if not (
        aggregate.get("run_id") == V1_RUN_ID
        and aggregate.get("status") == "completed"
        and aggregate.get("audit_opened") is True
        and aggregate.get("model_subset_searched_on_audit") is False
        and aggregate.get("public_code_copied") is False
        and aggregate.get("public_predictions_copied") is False
        and aggregate.get("public_leaderboard_used_for_selection") is False
        and len(members) == EXPECTED_MEMBER_COUNT
        and len({row["member"] for row in members}) == EXPECTED_MEMBER_COUNT
    ):
        raise RuntimeError("Recovered graph-context aggregate is ineligible")
    return aggregate, members


def evaluate(
    *,
    data_root: Path,
    models_root: Path,
    batch_size: int,
    threads: int,
    audit_only: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0 or threads <= 0:
        raise ValueError("batch size and thread count must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    manifest_path = data_root / "graph_context_relational_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    aggregate, members = _load_members(models_root)
    selection = (
        None
        if audit_only
        else eligible_subset(load_role(data_root, manifest, "selection"))
    )
    audit = eligible_subset(load_role(data_root, manifest, "audit"))
    roles = [("audit", audit)]
    if selection is not None:
        roles.insert(0, ("selection", selection))
    scores: dict[str, dict[str, torch.Tensor]] = {role: {} for role, _ in roles}
    started = time.monotonic()
    for row in members:
        member_started = time.monotonic()
        print(f"CPU diagnostic loading {row['member']}", flush=True)
        model = GraphContextDivisionModel().cpu()
        state = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        for role, data in roles:
            scores[role][row["member"]] = predict_cpu(
                model, data, batch_size=batch_size
            )
        del state, model
        gc.collect()
        print(
            f"CPU diagnostic completed {row['member']} in "
            f"{time.monotonic() - member_started:.3f}s",
            flush=True,
        )
    result_metrics = {}
    for role, data in roles:
        ensemble = calibration_free_equal_rank_ensemble(
            [scores[role][row["member"]] for row in members]
        )
        result_metrics[role] = eligible_metrics(data[4], ensemble, data[6], data[7])
    selection_reference = aggregate["selection_ensemble"]
    selection_ap_delta = None
    if selection is not None:
        selection_ap_delta = (
            float(result_metrics["selection"]["average_precision"])
            - float(selection_reference["average_precision"])
        )
    return {
        "schema_version": 1,
        "run_id": "graph-context-v1-all-member-cpu-diagnostic-v1",
        "status": "completed_diagnostic_only",
        "source_run_id": V1_RUN_ID,
        "member_policy": "all_selection_admitted_equal_rank",
        "member_count": len(members),
        "members": [
            {"member": row["member"], "model_sha256": row["model_sha256"]}
            for row in members
        ],
        "metrics": result_metrics,
        "audit_policy_shape_passed": passes_selection_gate(result_metrics["audit"]),
        "selection_reference_average_precision": float(
            selection_reference["average_precision"]
        ),
        "selection_cpu_float32_average_precision_delta": selection_ap_delta,
        "selection_cpu_reproduction_skipped": audit_only,
        "cpu_float32_diagnostic": True,
        "inference_eligible_rows_only": True,
        "audit_rows_scored": len(audit[0]),
        "audit_was_already_opened_by_source_run": True,
        "authorized_for_submission": False,
        "authorized_for_model_selection": False,
        "requires_fresh_seed_v2_and_complete_movie_gate": True,
        "competition_test_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(
        data_root=args.data_root,
        models_root=args.models_root,
        batch_size=args.batch_size,
        threads=args.threads,
        audit_only=args.audit_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
