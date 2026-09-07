#!/usr/bin/env python
"""Expand train-only detector adaptation frames without opening held-out roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.division_recovery_feasibility import graph_plain


RUN_ID = "competition-real-localization-expanded-inventory-v2"
PARENT_RUN_ID = "competition-real-localization-inventory-v1"
PARENT_SHA256 = "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1"
MAXIMUM_OPTIMIZATION_CENTERS = 5
FINAL_PROBE_STEMS = {
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def temporally_spread_centers(
    eligible: list[int], existing: list[int], *, maximum: int
) -> list[int]:
    """Preserve event centers, then greedily maximize temporal coverage."""

    available = sorted(set(int(value) for value in eligible))
    selected = sorted(set(int(value) for value in existing))
    if not set(selected).issubset(available):
        raise ValueError("existing center is not an eligible annotated frame")
    target_count = max(len(selected), min(maximum, len(available)))
    while len(selected) < target_count:
        remaining = [value for value in available if value not in selected]
        candidate = max(
            remaining,
            key=lambda value: (
                min(abs(value - chosen) for chosen in selected)
                if selected
                else float("inf"),
                -value,
            ),
        )
        selected.append(candidate)
        selected.sort()
    return selected


def resolve_geff(root: Path, stem: str) -> Path:
    candidates = (root / f"{stem}.geff", root / "train" / f"{stem}.geff")
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one GEFF for {stem}, saw {matches}")
    return matches[0]


def expand_movies(
    movies: list[dict[str, Any]],
    geff_root: Path,
    *,
    graph_loader: Callable[[Path], tuple[dict[int, tuple[float, ...]], Any]] = graph_plain,
) -> list[dict[str, Any]]:
    expanded = []
    for source in movies:
        movie = dict(source)
        stem = str(movie["stem"])
        role = str(movie["role"])
        if stem in FINAL_PROBE_STEMS:
            raise ValueError("final probe entered expanded detector inventory")
        nodes, _edges = graph_loader(resolve_geff(geff_root, stem))
        times = [int(values[0]) for values in nodes.values()]
        maximum_time = max(times)
        eligible = sorted({value for value in times if 0 < value < maximum_time})
        original = [int(value) for value in movie["center_frames"]]
        centers = (
            temporally_spread_centers(
                eligible, original, maximum=MAXIMUM_OPTIMIZATION_CENTERS
            )
            if role == "optimization"
            else sorted(set(original))
        )
        if role != "optimization" and centers != sorted(set(original)):
            raise RuntimeError("held-out center inventory changed")
        movie["center_frames"] = centers
        movie["required_frames"] = sorted(
            {center + offset for center in centers for offset in (-1, 0, 1)}
        )
        movie["annotated_nodes_at_centers"] = sum(value in centers for value in times)
        movie["expanded_optimization_centers"] = len(centers) - len(set(original))
        expanded.append(movie)
    return sorted(expanded, key=lambda row: (row["role"], row["stem"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-inventory", type=Path, required=True)
    parser.add_argument("--geff-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.parent_inventory) != PARENT_SHA256:
        raise ValueError("parent real-localization inventory changed")
    parent = json.loads(args.parent_inventory.read_text(encoding="utf-8"))
    movies = parent.get("movies")
    if not (
        parent.get("schema_version") == 1
        and parent.get("status") == "complete"
        and parent.get("run_id") == PARENT_RUN_ID
        and parent.get("competition_train_data_read") is True
        and parent.get("competition_test_data_read") is False
        and parent.get("public_leaderboard_used_for_selection") is False
        and parent.get("authorized_for_submission") is False
        and isinstance(movies, list)
        and len(movies) == 115
    ):
        raise ValueError("parent real-localization inventory is ineligible")
    records = expand_movies(movies, args.geff_root)
    by_role = {
        role: {
            "movies": sum(row["role"] == role for row in records),
            "center_frames": sum(
                len(row["center_frames"]) for row in records if row["role"] == role
            ),
            "annotated_nodes_at_centers": sum(
                int(row["annotated_nodes_at_centers"])
                for row in records
                if row["role"] == role
            ),
        }
        for role in ("optimization", "selection", "sealed_audit")
    }
    parent_by_role = parent["summary"]["by_role"]
    if not (
        by_role["optimization"]["movies"] == 96
        and by_role["selection"]["movies"] == 10
        and by_role["sealed_audit"]["movies"] == 9
        and by_role["optimization"]["center_frames"]
        > int(parent_by_role["optimization"]["center_frames"])
        and by_role["selection"]["center_frames"]
        == int(parent_by_role["selection"]["center_frames"])
        and by_role["sealed_audit"]["center_frames"]
        == int(parent_by_role["sealed_audit"]["center_frames"])
    ):
        raise RuntimeError("expanded real-localization role inventory changed")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "run_id": RUN_ID,
        "parent_inventory_sha256": PARENT_SHA256,
        "expansion_policy": {
            "roles_expanded": ["optimization"],
            "maximum_centers_per_optimization_movie": MAXIMUM_OPTIMIZATION_CENTERS,
            "method": "preserve_event_centers_then_greedy_maximum_temporal_separation",
            "selection_centers_changed": False,
            "sealed_audit_centers_changed": False,
        },
        "movies": records,
        "summary": {
            "movies": len(records),
            "center_frames": sum(len(row["center_frames"]) for row in records),
            "required_frames": sum(len(row["required_frames"]) for row in records),
            "by_role": by_role,
        },
        "excluded_final_probe_stems": sorted(FINAL_PROBE_STEMS),
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
