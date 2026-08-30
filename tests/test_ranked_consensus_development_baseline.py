from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify-ranked-consensus-development-baseline.py"
MODULE = runpy.run_path(str(SCRIPT))
STEMS = MODULE["EXPECTED_STEMS"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def make_sources(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    prediction = tmp_path / "prediction"
    truth = tmp_path / "truth"
    for root, prefix in ((prediction, "prediction"), (truth, "truth")):
        for stem in STEMS:
            graph = root / f"{stem}.geff"
            graph.mkdir(parents=True)
            (graph / "graph.txt").write_text(f"{prefix}-{stem}", encoding="utf-8")
    deep = tmp_path / "deep.json"
    morphology = tmp_path / "morphology.json"
    evaluator = tmp_path / "evaluator.py"
    deep.write_text("deep-probe", encoding="utf-8")
    morphology.write_text("morphology-probe", encoding="utf-8")
    evaluator.write_text("evaluator-source", encoding="utf-8")
    archived = tmp_path / "archived.json"
    replay = tmp_path / "replay.json"
    pooled = {
        "edge_before": {"fn": 2, "fp": 1, "jaccard": 0.7, "tp": 7},
        "edge_after": {"fn": 1, "fp": 1, "jaccard": 0.8, "tp": 8},
        "division_before": {"fn": 1, "fp": 0, "jaccard": 0.0, "tp": 0},
        "division_after": {"fn": 0, "fp": 0, "jaccard": 1.0, "tp": 1},
    }
    evidence = {
        "status": "development_positive",
        "selected": 1,
        "tp": 1,
        "fp": 0,
        "pooled": pooled,
        "authorized_for_full_candidate_evaluation": True,
        "authorized_for_submission": False,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    write_json(archived, evidence)
    replay_payload = copy.deepcopy(evidence)
    replay_payload["deep_probe_sha256"] = sha256(deep)
    replay_payload["morphology_probe_sha256"] = sha256(morphology)
    write_json(replay, replay_payload)
    paths = {
        "prediction_root": prediction,
        "truth_root": truth,
        "deep_probe": deep,
        "morphology_probe": morphology,
        "evaluator_source": evaluator,
        "archived_evidence": archived,
        "replay_evidence": replay,
    }
    contract = {
        "prediction_artifacts": MODULE["graph_inventory"](prediction),
        "truth_artifacts": MODULE["graph_inventory"](truth),
        "deep_probe_sha256": sha256(deep),
        "morphology_probe_sha256": sha256(morphology),
        "evaluator_source_sha256": sha256(evaluator),
        "archived_evidence_sha256": sha256(archived),
        "replay_evidence_sha256": sha256(replay),
        "selected": 1,
        "tp": 1,
        "fp": 0,
        "pooled": pooled,
    }
    return paths, contract


def test_verifier_binds_exact_graphs_probes_and_substantive_replay(tmp_path: Path) -> None:
    paths, contract = make_sources(tmp_path)

    result = MODULE["verify_baseline"](**paths, contract=contract)

    assert result["status"] == "verified"
    assert result["processed_public_control_cache_accepted"] is False
    assert result["authorized_for_submission"] is False
    assert result["prediction_artifacts"] == contract["prediction_artifacts"]


def test_verifier_rejects_graph_drift(tmp_path: Path) -> None:
    paths, contract = make_sources(tmp_path)
    graph = paths["prediction_root"] / f"{STEMS[0]}.geff" / "graph.txt"
    graph.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="prediction artifact inventory changed"):
        MODULE["verify_baseline"](**paths, contract=contract)


def test_verifier_rejects_substantive_replay_drift(tmp_path: Path) -> None:
    paths, contract = make_sources(tmp_path)
    replay = json.loads(paths["replay_evidence"].read_text(encoding="utf-8"))
    replay["fp"] = 1
    write_json(paths["replay_evidence"], replay)
    contract["replay_evidence_sha256"] = sha256(paths["replay_evidence"])

    with pytest.raises(ValueError, match="differs substantively"):
        MODULE["verify_baseline"](**paths, contract=contract)
