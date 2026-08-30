#!/usr/bin/env python
"""Verify and describe the exact ranked-consensus development baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biohub_tracker.graphs import artifact_tree_sha256


RUN_ID = "competition-ranked-consensus-development-baseline-v1"
EXPECTED_STEMS = (
    "44b6_12dfb391",
    "44b6_267148e4",
    "6bba_062c8d37",
    "6bba_07e24132",
)
FROZEN_CONTRACT: dict[str, Any] = {
    "prediction_artifacts": {
        "44b6_12dfb391": "91d892142c37974af1e336e5b2abc30afe553b029b5587853beadeb55cc42224",
        "44b6_267148e4": "a5a4ef83e8a1d5db87cb1a26313c2863a67b192e3323fa13ccb00614d4b609ad",
        "6bba_062c8d37": "c617b736f0a860ef0223bf8f6c572936c68ea1c83c3fdfd6ed46e5f00bc6167a",
        "6bba_07e24132": "f80a38be06bd1dfd3561e81b2d2f829d0675f12e97f2e3f29369fe9361287b25",
    },
    "truth_artifacts": {
        "44b6_12dfb391": "3ab99b7745872f50d0f55532c7284a7e648c617bfa9cf4bf203af6387b2c41ca",
        "44b6_267148e4": "983acc7565e54cb51693e8bd855a23836168bbbe241f49828d838cf3d0712eab",
        "6bba_062c8d37": "c7dc6c4178cd63ef5abb77e852bd048ce515b0313434bfb07e6188094fc66048",
        "6bba_07e24132": "b0bf2e5c4380c972f062ff8361ce8ec2523730d6d2257b52df66793bd435c456",
    },
    "deep_probe_sha256": "eb6eda5c42ef9d923e87008726c2147cd6b6105aeb5ffa364bac70e4e5f51b70",
    "morphology_probe_sha256": "729df1cb9484504a50162da1219aa166d22403cdc117f99300d837a0dfe944f1",
    "evaluator_source_sha256": "e02dafca22b39866d10aa8db44f8e6872d352514d5671a25746d0161070523a9",
    "archived_evidence_sha256": "ffe9f652c07cd9ebf37ab2b6e84b9131fa93001f33680dbc985de6a089988606",
    "replay_evidence_sha256": "5b9942f8ec31bfce7631cfb68acf8ed0f64d56922a7695059e46ac22297c24bd",
    "pooled": {
        "edge_before": {"fn": 106, "fp": 85, "jaccard": 0.9196804037005887, "tp": 2187},
        "edge_after": {"fn": 103, "fp": 85, "jaccard": 0.9209419680403701, "tp": 2190},
        "division_before": {"fn": 5, "fp": 0, "jaccard": 0.0, "tp": 0},
        "division_after": {"fn": 2, "fp": 0, "jaccard": 0.6, "tp": 3},
    },
    "selected": 3,
    "tp": 3,
    "fp": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def graph_inventory(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"graph root is missing: {root}")
    graphs = sorted(path for path in root.glob("*.geff") if path.is_dir())
    inventory = {path.stem: artifact_tree_sha256(path) for path in graphs}
    if tuple(inventory) != EXPECTED_STEMS:
        raise ValueError(
            f"graph inventory changed for {root}: expected {EXPECTED_STEMS}, "
            f"observed {tuple(inventory)}"
        )
    return inventory


def substantive_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result.pop("deep_probe_sha256", None)
    result.pop("morphology_probe_sha256", None)
    return result


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} changed: expected {expected}, observed {observed}")
    return observed


def verify_baseline(
    *,
    prediction_root: Path,
    truth_root: Path,
    deep_probe: Path,
    morphology_probe: Path,
    evaluator_source: Path,
    archived_evidence: Path,
    replay_evidence: Path,
    contract: Mapping[str, Any] = FROZEN_CONTRACT,
) -> dict[str, Any]:
    prediction_inventory = graph_inventory(prediction_root)
    truth_inventory = graph_inventory(truth_root)
    if prediction_inventory != contract["prediction_artifacts"]:
        raise ValueError("EMA 0.940 prediction artifact inventory changed")
    if truth_inventory != contract["truth_artifacts"]:
        raise ValueError("development truth artifact inventory changed")

    deep_hash = require_hash(deep_probe, contract["deep_probe_sha256"], "deep probe")
    morphology_hash = require_hash(
        morphology_probe,
        contract["morphology_probe_sha256"],
        "morphology probe",
    )
    evaluator_hash = require_hash(
        evaluator_source,
        contract["evaluator_source_sha256"],
        "ranked-consensus evaluator",
    )
    archived_hash = require_hash(
        archived_evidence,
        contract["archived_evidence_sha256"],
        "archived development evidence",
    )
    replay_hash = require_hash(
        replay_evidence,
        contract["replay_evidence_sha256"],
        "replayed development evidence",
    )

    archived = load_json(archived_evidence)
    replay = load_json(replay_evidence)
    if substantive_evidence(archived) != substantive_evidence(replay):
        raise ValueError("replay differs substantively from archived development evidence")
    if replay.get("deep_probe_sha256") != deep_hash:
        raise ValueError("replay is not bound to the frozen deep probe")
    if replay.get("morphology_probe_sha256") != morphology_hash:
        raise ValueError("replay is not bound to the frozen morphology probe")
    for field in ("selected", "tp", "fp", "pooled"):
        if replay.get(field) != contract[field]:
            raise ValueError(f"frozen development result changed: {field}")
    if not (
        replay.get("status") == "development_positive"
        and replay.get("authorized_for_full_candidate_evaluation") is True
        and replay.get("authorized_for_submission") is False
        and replay.get("competition_test_data_read") is False
        and replay.get("public_leaderboard_used_for_selection") is False
        and replay.get("submission_created") is False
    ):
        raise ValueError("frozen development evidence has unsafe authorization metadata")

    return {
        "schema_version": 1,
        "status": "verified",
        "run_id": RUN_ID,
        "baseline_family": "biohub-ct-0940-ema",
        "stems": list(EXPECTED_STEMS),
        "prediction_artifacts": prediction_inventory,
        "truth_artifacts": truth_inventory,
        "deep_probe_sha256": deep_hash,
        "morphology_probe_sha256": morphology_hash,
        "evaluator_source_sha256": evaluator_hash,
        "archived_evidence_sha256": archived_hash,
        "replay_evidence_sha256": replay_hash,
        "selected": replay["selected"],
        "tp": replay["tp"],
        "fp": replay["fp"],
        "pooled": replay["pooled"],
        "processed_public_control_cache_accepted": False,
        "processed_public_control_rejection_reason": (
            "its graph state already contains the frozen 44b6_267148e4/646->840 "
            "recovery edge, so it is not the pre-addition development baseline"
        ),
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--deep-probe", type=Path, required=True)
    parser.add_argument("--morphology-probe", type=Path, required=True)
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--archived-evidence", type=Path, required=True)
    parser.add_argument("--replay-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_baseline(
        prediction_root=args.prediction_root,
        truth_root=args.truth_root,
        deep_probe=args.deep_probe,
        morphology_probe=args.morphology_probe,
        evaluator_source=args.evaluator_source,
        archived_evidence=args.archived_evidence,
        replay_evidence=args.replay_evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
