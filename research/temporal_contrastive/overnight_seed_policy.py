"""Shared immutable member-selection policy for the overnight seed sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research.temporal_contrastive.evaluate_real_division_seed_ensemble import (
    REFERENCE_SELECTION_AP,
    RUN_ID as ENSEMBLE_RUN_ID,
    individual_admitted,
    sha256_file,
    stronger_than_reference,
)


ELIGIBLE_STATUSES = {
    "ensemble_eligible_for_development_probe",
    "stronger_individuals_eligible_for_development_probe",
}


def candidate_identity(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["seed"]), str(row["fold"]), str(row["model_sha256"])


def candidate_is_admitted(row: dict[str, Any]) -> bool:
    return bool(
        row.get("status") == "admitted"
        and individual_admitted(
            row.get("selection", {}),
            row.get("selection_by_embryo", {}),
            row.get("selection_frozen_threshold_diagnostic"),
        )
    )


def select_precommitted_members(
    terminal: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Return the one policy authorized before the development probe opens."""
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("run_id") == ENSEMBLE_RUN_ID
        and terminal.get("status") in ELIGIBLE_STATUSES
        and terminal.get("authorized_for_development_probe") is True
        and terminal.get("competition_test_data_read") is False
        and terminal.get("final_probe_opened") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise ValueError("overnight ensemble terminal is ineligible for the probe")

    admitted = [
        row for row in terminal.get("candidates", []) if candidate_is_admitted(row)
    ]
    identities = [candidate_identity(row) for row in admitted]
    if len(identities) != len(set(identities)):
        raise ValueError("admitted overnight member identities are not unique")
    hashes = [identity[2] for identity in identities]
    if len(hashes) != len(set(hashes)):
        raise ValueError("byte-identical checkpoints cannot both vote")

    if terminal.get("ensemble_eligible_for_development_probe") is True:
        ensemble = terminal.get("ensemble") or {}
        if not (
            terminal.get("status") == "ensemble_eligible_for_development_probe"
            and len(admitted) >= 2
            and ensemble.get("model_count") == len(admitted)
            and ensemble.get("absolute_threshold_authorized") is False
            and ensemble.get("policy")
            == "equal average of within-movie percentile ranks from every independently admitted model"
        ):
            raise ValueError("authorized equal-rank ensemble changed")
        return "equal_rank_admitted_ensemble", admitted

    stronger = [row for row in admitted if stronger_than_reference(row)]
    stronger.sort(
        key=lambda row: (
            -float(row["selection"]["average_precision"]),
            int(row["seed"]),
            str(row["fold"]),
        )
    )
    declared = terminal.get("stronger_individuals", [])
    if not (
        terminal.get("status")
        == "stronger_individuals_eligible_for_development_probe"
        and terminal.get("ensemble_eligible_for_development_probe") is False
        and stronger
        and declared
        and candidate_identity(stronger[0])
        == (
            int(declared[0]["seed"]),
            str(declared[0]["fold"]),
            str(declared[0]["model_sha256"]),
        )
    ):
        raise ValueError("strongest authorized individual changed")
    return "strongest_individual_rank", [stronger[0]]


def resolve_member_paths(
    members: list[dict[str, Any]], sweep_root: Path
) -> list[Path]:
    paths = []
    for member in members:
        path = (
            sweep_root
            / f"seed-{int(member['seed'])}"
            / str(member["fold"])
            / "division_model.pt"
        )
        if not path.is_file() or sha256_file(path) != member["model_sha256"]:
            raise ValueError(f"overnight checkpoint changed: {path}")
        paths.append(path)
    return paths
