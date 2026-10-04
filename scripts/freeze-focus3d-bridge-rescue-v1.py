"""Freeze a conservative FOCUS/DeepCenter single-frame bridge experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "research/graph_context_fresh_split_v3.json"
OUTPUT = ROOT / "research/focus3d_bridge_rescue_v1_policy.json"
SALT = "biohub-focus3d-bridge-rescue-v1"
EXCLUDED_STEMS = {
    "44b6_d754aa59", "44b6_7a302da0", "6bba_debd7bfa", "6bba_fc5f39dc",
    "44b6_12dfb391", "44b6_267148e4", "6bba_062c8d37", "6bba_07e24132",
}


def stable_key(embryo: str, stem: str) -> str:
    return hashlib.sha256(f"{SALT}\0{embryo}\0{stem}".encode()).hexdigest()


def freeze(inventory_path: Path = INVENTORY) -> dict[str, Any]:
    source = json.loads(inventory_path.read_text(encoding="utf-8"))
    by_embryo: dict[str, set[str]] = {"44b6": set(), "6bba": set()}
    for row in source["stem_roles"]:
        embryo, stem = str(row["embryo"]), str(row["stem"])
        if embryo in by_embryo and stem not in EXCLUDED_STEMS:
            by_embryo[embryo].add(stem)
    validation = []
    for embryo in ("44b6", "6bba"):
        ranked = sorted(by_embryo[embryo], key=lambda stem: stable_key(embryo, stem))
        validation.extend(ranked[:2])
    return {
        "schema_version": 1,
        "run_id": "focus3d-bridge-rescue-v1",
        "status": "frozen_before_focus_or_base_predictions_on_validation_stems",
        "hypothesis": "A FOCUS detection confirmed by DeepCenter can safely fill one missing single-frame node between two unlinked base endpoints.",
        "candidate_base": "redoctopusk/biohub-948tta2",
        "focus3d_role": "detector proposal only; physical-linker edges are local support, never copied into the candidate",
        "policy": {
            "maximum_added_nodes_per_movie": 1,
            "maximum_added_edges_per_movie": 2,
            "gap_frames": 2,
            "base_endpoint_total_distance_max_um": 10.0,
            "base_endpoint_step_max_um": 5.0,
            "focus_to_linear_interpolation_max_um": 1.5,
            "base_same_frame_exclusion_radius_um": 3.0,
            "deepcenter_minimum_probability": 0.25,
            "require_focus_incoming_and_outgoing_support": True,
            "require_base_predecessor_out_degree_zero": True,
            "require_base_successor_in_degree_zero": True,
            "forbid_division_endpoint_mutation": True,
            "candidate_ranking": "minimum physical interpolation residual, then node id",
        },
        "validation_stems": validation,
        "validation_selection": {
            "salt": SALT,
            "rule": "first two SHA-256 ordered non-prior-evaluation stems per embryo",
            "labels_used": False,
            "model_outputs_used": False,
            "leaderboard_used": False,
            "excluded_prior_evaluation_stems": sorted(EXCLUDED_STEMS),
        },
        "promotion_gate": {
            "complete_movie_count": 4,
            "patched_proxy_delta_strictly_positive": True,
            "minimum_movie_adjusted_edge_delta": 0.0,
            "division_jaccard_must_not_decrease": True,
            "node_count_penalty_must_not_increase": True,
            "production_must_add_at_least_one_bridge": True,
        },
        "metric_hack_used": False,
        "public_predictions_copied": False,
        "leaderboard_used_for_selection": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    payload = freeze()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
