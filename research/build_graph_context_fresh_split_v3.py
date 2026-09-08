#!/usr/bin/env python
"""Freeze a new movie-disjoint split for graph-context division training.

The assignment is a pure function of embryo, movie stem, and a fixed salt.  It
does not use labels, model outputs, leaderboard results, or the prior role.  We
compute label counts only after the mapping is frozen so an unusable partition
fails closed instead of being reshuffled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RUN_ID = "competition-graph-context-fresh-split-v3"
SOURCE_RUN_ID = "competition-graph-context-relational-patches-v1"
SPLIT_SALT = "biohub-graph-context-fresh-split-v3"
ROLE_ORDER = ("audit", "selection", "optimization")
EXPECTED_SOURCE_SUMMARY = {
    "movies": 146,
    "rows": 3013,
    "positives": 134,
    "hard_negatives": 2879,
    "inference_eligible_positives": 55,
    "inference_eligible_hard_negatives": 165,
}
EXPECTED_STEMS_BY_EMBRYO = {"44b6": 36, "6bba": 110}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(*, embryo: str, stem: str) -> str:
    material = f"{SPLIT_SALT}\0{embryo}\0{stem}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def assign_roles(stems_by_embryo: dict[str, list[str]]) -> dict[str, str]:
    """Assign 20% audit, 20% selection, and the remainder optimization."""

    assigned: dict[str, str] = {}
    for embryo in ("44b6", "6bba"):
        stems = sorted(
            set(stems_by_embryo.get(embryo, [])),
            key=lambda stem: (stable_key(embryo=embryo, stem=stem), stem),
        )
        if len(stems) != EXPECTED_STEMS_BY_EMBRYO[embryo]:
            raise ValueError(f"unexpected {embryo} movie inventory: {len(stems)}")
        holdout_count = int(math.ceil(0.20 * len(stems)))
        sau = stems[:holdout_count]
        selection = stems[holdout_count : 2 * holdout_count]
        optimization = stems[2 * holdout_count :]
        for role, members in zip(
            ROLE_ORDER, (sau, selection, optimization), strict=True
        ):
            for stem in members:
                if stem in assigned:
                    raise RuntimeError(f"duplicate stem in split: {stem}")
                assigned[stem] = role
    return assigned


def validate_source(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("records")
    summary = manifest.get("summary", {})
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == SOURCE_RUN_ID
        and isinstance(records, list)
        and len(records) == 2274
        and all(summary.get(key) == value for key, value in EXPECTED_SOURCE_SUMMARY.items())
        and manifest.get("audit_labels_scored") is False
        and manifest.get("final_probe_movies_extracted") is False
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("authorized_for_submission") is False
    ):
        raise ValueError("graph-context source manifest is ineligible")
    return records


def build_split(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = validate_source(manifest)
    stems_by_embryo = {
        embryo: sorted(
            {
                str(record["stem"])
                for record in records
                if record.get("embryo") == embryo
            }
        )
        for embryo in ("44b6", "6bba")
    }
    roles = assign_roles(stems_by_embryo)
    if set(roles) != {str(record["stem"]) for record in records}:
        raise RuntimeError("fresh split does not cover the exact source inventory")

    summaries: dict[str, dict[str, dict[str, int]]] = {}
    for embryo in ("44b6", "6bba"):
        summaries[embryo] = {}
        for role in ROLE_ORDER:
            selected = [
                record
                for record in records
                if record.get("embryo") == embryo
                and roles[str(record["stem"])] == role
            ]
            summaries[embryo][role] = {
                "movies": len({str(row["stem"]) for row in selected}),
                "records": len(selected),
                "rows": sum(int(row["rows"]) for row in selected),
                "positives": sum(int(row["positives"]) for row in selected),
                "hard_negatives": sum(int(row["hard_negatives"]) for row in selected),
                "inference_eligible_positives": sum(
                    int(row["inference_eligible_positives"]) for row in selected
                ),
                "inference_eligible_hard_negatives": sum(
                    int(row["inference_eligible_hard_negatives"]) for row in selected
                ),
            }
    for embryo in summaries.values():
        for role in ROLE_ORDER:
            row = embryo[role]
            if not (
                row["movies"] > 0
                and row["positives"] > 0
                and row["hard_negatives"] > 0
                and row["inference_eligible_positives"] > 0
                and row["inference_eligible_hard_negatives"] > 0
            ):
                raise RuntimeError(f"fresh split produced an unusable stratum: {role}")

    validation_stems: list[str] = []
    for embryo in ("44b6", "6bba"):
        eligible_stems = sorted(
            {
                str(record["stem"])
                for record in records
                if record.get("embryo") == embryo
                and roles[str(record["stem"])] == "audit"
                and int(record["inference_eligible_positives"]) > 0
            },
            key=lambda stem: (stable_key(embryo=embryo, stem=stem), stem),
        )
        if len(eligible_stems) < 2:
            raise RuntimeError(f"fresh audit lacks two event movies for {embryo}")
        validation_stems.extend(eligible_stems[:2])

    return {
        "schema_version": 1,
        "status": "frozen_before_model_scoring",
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "source_manifest_sha256": sha256_file(manifest_path),
        "split_salt": SPLIT_SALT,
        "split_rule": (
            "within each embryo, sort unique stems by "
            "sha256(salt\\0embryo\\0stem); assign the first ceil(20%) to "
            "audit, the next ceil(20%) to selection, and the remainder to "
            "optimization"
        ),
        "labels_used_for_assignment": False,
        "labels_used_only_to_require_validation_event_presence": True,
        "model_outputs_used_for_assignment": False,
        "prior_roles_used_for_assignment": False,
        "leaderboard_used_for_assignment": False,
        "stem_roles": [
            {
                "stem": stem,
                "embryo": stem.split("_", 1)[0],
                "role": roles[stem],
                "stable_key": stable_key(
                    embryo=stem.split("_", 1)[0], stem=stem
                ),
            }
            for stem in sorted(roles)
        ],
        "summary": summaries,
        "complete_movie_validation_stems": validation_stems,
        "complete_movie_validation_rule": (
            "for each embryo, take the first two fresh-audit stems by the "
            "already-frozen stable key among stems containing at least one "
            "inference-eligible positive; do not use model scores or prior "
            "per-movie performance"
        ),
        "frozen_experiment": {
            "family": "temporal_multiscale_graph_context_division_v1",
            "parameter_count_per_member": 74732308,
            "external_only_warm_starts": {
                "target_44b6": "a0a1794134893d191d896b8b83abee61a754bc3852e2b8f74dacd353ce118a76",
                "target_6bba": "9e8af9aeb247297d3ed09bda3bcc2b5af413d78c6bf2546eff5f4898667e074d",
            },
            "warm_start_competition_data_read": False,
            "seeds": [1409101, 1509107],
            "planned_model_count": 4,
            "steps_per_model": 20000,
            "batch_size": 10,
            "validation_batch_size": 20,
            "validation_every": 250,
            "learning_rate": 0.00006,
            "minimum_learning_rate": 0.0000003,
            "backbone_learning_rate_multiplier": 0.15,
            "weight_decay": 0.0002,
            "ema_decay": 0.995,
            "selection_policy": (
                "admit each member only at AP>=0.55, per-embryo AP>=0.40, "
                "and at least two true positives before its first false "
                "positive; freeze the equal-rank ensemble of every admitted "
                "member before opening audit"
            ),
            "independent_voter": (
                "132-feature temporal morphology ensemble fitted only on "
                "fresh optimization movies"
            ),
            "independent_voter_selection_gate": (
                "pooled eligible AP>=0.55, per-embryo eligible AP>=0.40, and "
                "at least two eligible true positives before the first false "
                "positive"
            ),
            "deployment_rule": (
                "within each movie, add at most one parent-free edge only "
                "when the graph-context equal-rank ensemble and independent "
                "morphology ranking select the same geometry>=3 candidate"
            ),
            "absolute_threshold_for_deployment": False,
            "audit_gate": (
                "at least three true agreed recoveries, at most one false "
                "agreed recovery, positive pooled eligible Jaccard delta, "
                "and no embryo with zero true agreed recoveries"
            ),
            "complete_movie_promotion_gate": (
                "patched-official pooled proxy delta>0, pooled adjusted-edge "
                "delta>=0, division Jaccard strictly improves, no movie proxy "
                "regresses, graph integrity passes, and production differs "
                "from the attributed public control"
            ),
            "candidate_base": "redoctopusk/biohub-948tta2",
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "metric_hack_allowed": False,
        },
        "audit_model_predictions_generated": False,
        "audit_scores_opened": False,
        "competition_test_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_split(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
