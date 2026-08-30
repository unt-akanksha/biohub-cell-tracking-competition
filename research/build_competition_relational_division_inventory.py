#!/usr/bin/env python
"""Build candidate-aligned relational division examples from official train GEFFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.build_competition_division_training_inventory import (
    FINAL_PROBE_STEMS,
    atomic_json,
    validate_geff_cache,
)
from research.division_recovery_feasibility import graph_plain
from research.learned_division_recovery import (
    VOXEL_SIZE_ZYX_UM,
    biological_geometry_score,
)


RUN_ID = "competition-relational-division-inventory-v3"
ROLE_FRACTION = 0.20
TRAINING_GEOMETRY_MINIMUM = 0.0
INFERENCE_GEOMETRY_MINIMUM = 3.0
PARENT_DISTANCE_MAX_UM = 12.0
SISTER_DISTANCE_MAX_UM = 15.0
MAXIMUM_NEGATIVES_PER_PARENT = 2
MINIMUM_NEGATIVES_PER_MOVIE = 32
NEGATIVES_PER_POSITIVE = 16


def stable_key(value: str, namespace: str) -> str:
    return hashlib.sha256(f"{RUN_ID}|{namespace}|{value}".encode()).hexdigest()


def select_negative_examples(
    stem: str, rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("relational negative limit must be nonnegative")
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["inference_geometry_eligible"]),
            stable_key(
                f"{stem}|{row['parent_id']}|{row['proposed_child_id']}",
                "negative",
            ),
            int(row["parent_id"]),
            int(row["proposed_child_id"]),
        ),
    )
    return ranked[:limit]


def physical(row: tuple[float, ...]) -> np.ndarray:
    return np.asarray(row[1:], dtype=np.float64) * VOXEL_SIZE_ZYX_UM


def relational_features(
    nodes: dict[int, tuple[float, ...]],
    incoming: dict[int, list[int]],
    parent_id: int,
    existing_child_id: int,
    proposed_child_id: int,
) -> dict[str, float | int | None]:
    parent = physical(nodes[parent_id])
    existing_step = physical(nodes[existing_child_id]) - parent
    proposed_step = physical(nodes[proposed_child_id]) - parent
    geometry = biological_geometry_score(existing_step, proposed_step)
    predecessors = [
        source
        for source in incoming.get(parent_id, ())
        if int(nodes[source][0]) == int(nodes[parent_id][0]) - 1
    ]
    predecessor_id = predecessors[0] if len(predecessors) == 1 else None
    velocity = (
        parent - physical(nodes[predecessor_id])
        if predecessor_id is not None
        else None
    )
    midpoint = parent + 0.5 * (existing_step + proposed_step)
    return {
        "parent_distance_um": float(np.linalg.norm(proposed_step)),
        "sister_distance_um": float(
            np.linalg.norm(physical(nodes[proposed_child_id]) - physical(nodes[existing_child_id]))
        ),
        "existing_distance_um": geometry[1],
        "daughter_midpoint_distance_um": geometry[2],
        "daughter_opposition_cosine": geometry[3],
        "daughter_step_ratio": geometry[4],
        "biological_geometry_score": geometry[0],
        "predecessor_id": predecessor_id,
        "parent_velocity_um": (
            float(np.linalg.norm(velocity)) if velocity is not None else None
        ),
        "constant_velocity_midpoint_error_um": (
            float(np.linalg.norm(midpoint - (parent + velocity)))
            if velocity is not None
            else None
        ),
    }


def example_record(
    stem: str,
    nodes: dict[int, tuple[float, ...]],
    incoming: dict[int, list[int]],
    *,
    parent_id: int,
    existing_child_id: int,
    proposed_child_id: int,
    target: bool,
) -> dict[str, Any]:
    timepoint = int(nodes[parent_id][0])
    if not (
        int(nodes[existing_child_id][0]) == timepoint + 1
        and int(nodes[proposed_child_id][0]) == timepoint + 1
    ):
        raise ValueError("relational daughters must be in the next frame")
    centers = {
        "parent": [float(value) for value in nodes[parent_id][1:]],
        "existing_child": [float(value) for value in nodes[existing_child_id][1:]],
        "proposed_child": [float(value) for value in nodes[proposed_child_id][1:]],
    }
    features = relational_features(
        nodes,
        incoming,
        parent_id,
        existing_child_id,
        proposed_child_id,
    )
    return {
        "stem": stem,
        "embryo": stem.split("_", 1)[0],
        "timepoint": timepoint,
        "required_frames": [max(0, timepoint - 1), timepoint, timepoint + 1],
        "parent_id": int(parent_id),
        "existing_child_id": int(existing_child_id),
        "proposed_child_id": int(proposed_child_id),
        "centers_zyx_voxel": centers,
        "division_recovery_target": bool(target),
        "daughter_order_invariant": True,
        "inference_geometry_eligible": bool(
            float(features["biological_geometry_score"])
            >= INFERENCE_GEOMETRY_MINIMUM
        ),
        **features,
    }


def movie_examples(
    stem: str,
    nodes: dict[int, tuple[float, ...]],
    edges: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    by_time: dict[int, list[int]] = {}
    for node_id, row in nodes.items():
        by_time.setdefault(int(row[0]), []).append(int(node_id))
    for source, target in edges:
        if source not in nodes or target not in nodes:
            raise ValueError("train GEFF contains a dangling edge")
        if int(nodes[target][0]) != int(nodes[source][0]) + 1:
            raise ValueError("train GEFF contains a non-adjacent edge")
        outgoing.setdefault(int(source), []).append(int(target))
        incoming.setdefault(int(target), []).append(int(source))

    positives: list[dict[str, Any]] = []
    negative_pool: list[dict[str, Any]] = []
    for parent_id, children in sorted(outgoing.items()):
        children = sorted(set(children))
        if len(children) not in (1, 2):
            continue
        timepoint = int(nodes[parent_id][0])
        next_nodes = by_time.get(timepoint + 1, ())
        existing_child_id = children[0]
        if len(children) == 2:
            positive = example_record(
                stem,
                nodes,
                incoming,
                parent_id=parent_id,
                existing_child_id=existing_child_id,
                proposed_child_id=children[1],
                target=True,
            )
            if (
                positive["parent_distance_um"] <= PARENT_DISTANCE_MAX_UM
                and positive["sister_distance_um"] <= SISTER_DISTANCE_MAX_UM
                and positive["biological_geometry_score"]
                >= TRAINING_GEOMETRY_MINIMUM
            ):
                positives.append(positive)

        candidates = []
        for proposed_child_id in next_nodes:
            if proposed_child_id in children:
                continue
            row = example_record(
                stem,
                nodes,
                incoming,
                parent_id=parent_id,
                existing_child_id=existing_child_id,
                proposed_child_id=proposed_child_id,
                target=False,
            )
            if (
                row["parent_distance_um"] <= PARENT_DISTANCE_MAX_UM
                and row["sister_distance_um"] <= SISTER_DISTANCE_MAX_UM
                and row["biological_geometry_score"]
                >= TRAINING_GEOMETRY_MINIMUM
            ):
                candidates.append(row)
        candidates.sort(
            key=lambda row: (
                -float(row["biological_geometry_score"]),
                float(row["parent_distance_um"]),
                float(row["sister_distance_um"]),
                int(row["proposed_child_id"]),
            )
        )
        negative_pool.extend(candidates[:MAXIMUM_NEGATIVES_PER_PARENT])

    negative_limit = max(
        MINIMUM_NEGATIVES_PER_MOVIE,
        len(positives) * NEGATIVES_PER_POSITIVE,
    )
    return [
        *positives,
        *select_negative_examples(stem, negative_pool, negative_limit),
    ]


def allocate_roles(movie_rows: list[dict[str, Any]]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for embryo in ("44b6", "6bba"):
        eligible = [
            movie
            for movie in movie_rows
            if movie["embryo"] == embryo
            and movie["stem"] not in FINAL_PROBE_STEMS
            and movie["inference_eligible_positives"] > 0
        ]
        target = int(
            math.ceil(
                sum(row["inference_eligible_positives"] for row in eligible)
                * ROLE_FRACTION
            )
        )
        audit: set[str] = set()
        accumulated = 0
        for row in sorted(
            eligible,
            key=lambda item: stable_key(item["stem"], f"{embryo}-audit"),
        ):
            audit.add(row["stem"])
            accumulated += int(row["inference_eligible_positives"])
            if accumulated >= target:
                break
        remaining = [row for row in eligible if row["stem"] not in audit]
        selection: set[str] = set()
        accumulated = 0
        for row in sorted(
            remaining,
            key=lambda item: stable_key(item["stem"], f"{embryo}-selection"),
        ):
            selection.add(row["stem"])
            accumulated += int(row["inference_eligible_positives"])
            if accumulated >= target:
                break
        negative_target = int(
            math.ceil(
                sum(
                    row["inference_eligible_hard_negatives"]
                    for row in movie_rows
                    if row["embryo"] == embryo
                    and row["stem"] not in FINAL_PROBE_STEMS
                )
                * ROLE_FRACTION
            )
        )
        negative_pool = [
            row
            for row in movie_rows
            if row["embryo"] == embryo
            and row["stem"] not in FINAL_PROBE_STEMS
            and row["stem"] not in audit
            and row["stem"] not in selection
            and row["inference_eligible_hard_negatives"] > 0
        ]
        for role_name, role_stems in (("audit", audit), ("selection", selection)):
            negative_rows = sum(
                row["inference_eligible_hard_negatives"]
                for row in movie_rows
                if row["stem"] in role_stems
            )
            for row in sorted(
                negative_pool,
                key=lambda item: stable_key(
                    item["stem"], f"{embryo}-{role_name}-negative"
                ),
            ):
                if negative_rows >= negative_target:
                    break
                if row["stem"] in audit or row["stem"] in selection:
                    continue
                role_stems.add(row["stem"])
                negative_rows += int(row["inference_eligible_hard_negatives"])
            if negative_rows < negative_target:
                raise RuntimeError(
                    f"relational {embryo} {role_name} lacks inference-eligible "
                    f"hard negatives: {negative_rows} < {negative_target}"
                )
        if not audit or not selection or audit & selection:
            raise RuntimeError(f"relational division split failed for {embryo}")
        for movie in movie_rows:
            if movie["embryo"] != embryo or movie["stem"] in FINAL_PROBE_STEMS:
                continue
            roles[movie["stem"]] = (
                "audit"
                if movie["stem"] in audit
                else "selection"
                if movie["stem"] in selection
                else "optimization"
            )
    return roles


def summarize(examples: list[dict[str, Any]], role: str, embryo: str) -> dict[str, int]:
    selected = [
        row for row in examples if row["role"] == role and row["embryo"] == embryo
    ]
    return {
        "movies": len({row["stem"] for row in selected}),
        "rows": len(selected),
        "positives": sum(bool(row["division_recovery_target"]) for row in selected),
        "hard_negatives": sum(
            not bool(row["division_recovery_target"]) for row in selected
        ),
        "inference_eligible_positives": sum(
            bool(row["division_recovery_target"])
            and bool(row["inference_geometry_eligible"])
            for row in selected
        ),
        "inference_eligible_hard_negatives": sum(
            not bool(row["division_recovery_target"])
            and bool(row["inference_geometry_eligible"])
            for row in selected
        ),
    }


def build_inventory(cache_root: Path) -> dict[str, Any]:
    manifest_path = cache_root / "train_geff_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stems = validate_geff_cache(cache_root, manifest)
    movie_payloads = []
    for index, stem in enumerate(stems, start=1):
        nodes, edges = graph_plain(cache_root / "train" / f"{stem}.geff")
        examples = movie_examples(stem, nodes, edges)
        movie_payloads.append(
            {
                "stem": stem,
                "embryo": stem.split("_", 1)[0],
                "examples": examples,
                "positives": sum(
                    bool(row["division_recovery_target"]) for row in examples
                ),
                "hard_negatives": sum(
                    not bool(row["division_recovery_target"]) for row in examples
                ),
                "inference_eligible_positives": sum(
                    bool(row["division_recovery_target"])
                    and bool(row["inference_geometry_eligible"])
                    for row in examples
                ),
                "inference_eligible_hard_negatives": sum(
                    not bool(row["division_recovery_target"])
                    and bool(row["inference_geometry_eligible"])
                    for row in examples
                ),
            }
        )
        if index % 25 == 0:
            print(f"built {index}/{len(stems)} relational movies", flush=True)
    roles = allocate_roles(movie_payloads)
    examples = []
    for movie in movie_payloads:
        if movie["stem"] in FINAL_PROBE_STEMS:
            continue
        role = roles[movie["stem"]]
        examples.extend({**row, "role": role} for row in movie["examples"])
    by_embryo_role = {
        embryo: {
            role: summarize(examples, role, embryo)
            for role in ("optimization", "selection", "audit")
        }
        for embryo in ("44b6", "6bba")
    }
    if not all(
        by_embryo_role[embryo][role]["positives"] > 0
        and by_embryo_role[embryo][role]["hard_negatives"] > 0
        for embryo in ("44b6", "6bba")
        for role in ("optimization", "selection", "audit")
    ):
        raise RuntimeError("relational split lost a class")
    result = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "policy": {
            "simulation": (
                "true division parent with one daughter treated as retained and the "
                "other as proposed; nearby non-child proposals are hard negatives"
            ),
            "daughter_order_invariant": True,
            "shared_temporal_frames": "t-1,t,t+1 at parent/existing/proposed centers",
            "training_geometry_minimum": TRAINING_GEOMETRY_MINIMUM,
            "inference_geometry_minimum": INFERENCE_GEOMETRY_MINIMUM,
            "parent_distance_max_um": PARENT_DISTANCE_MAX_UM,
            "sister_distance_max_um": SISTER_DISTANCE_MAX_UM,
            "maximum_negatives_per_parent": MAXIMUM_NEGATIVES_PER_PARENT,
            "negative_sampling": "deterministic SHA-256 within movie after geometry ranking",
            "role_fraction": ROLE_FRACTION,
        },
        "examples": examples,
        "summary": {
            "movies": len({row["stem"] for row in examples}),
            "rows": len(examples),
            "positives": sum(bool(row["division_recovery_target"]) for row in examples),
            "hard_negatives": sum(
                not bool(row["division_recovery_target"]) for row in examples
            ),
            "inference_eligible_positives": sum(
                bool(row["division_recovery_target"])
                and bool(row["inference_geometry_eligible"])
                for row in examples
            ),
            "inference_eligible_hard_negatives": sum(
                not bool(row["division_recovery_target"])
                and bool(row["inference_geometry_eligible"])
                for row in examples
            ),
            "by_embryo_role": by_embryo_role,
        },
        "source_geff_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "audit_opened": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_inventory(args.cache_root)
    atomic_json(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
