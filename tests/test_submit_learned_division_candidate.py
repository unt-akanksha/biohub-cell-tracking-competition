from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/submit-learned-division-candidate.py"


def test_validates_hash_bound_promoted_candidate(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,dataset,row_type\n", encoding="utf-8")
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "eligible_for_submission",
                "run_id": "ema-learned-division-candidate-v1",
                "target_public_score": 0.945,
                "known_public_hash_match": False,
                "learned_edges_added": 2,
                "learned_reassignments": 0,
                "learned_node_or_coordinate_changes": 0,
                "proxy_gain": 0.01,
                "competition_submission_performed": False,
                "authorized_for_submission": True,
                "submission_path": str(submission),
                "submission_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    module = runpy.run_path(str(SUBMITTER))

    payload, observed = module["validate_promotion"](promotion)

    assert observed == submission.resolve()
    assert payload["submission_sha256"] == digest


def test_rejects_candidate_whose_submission_changed(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("original", encoding="utf-8")
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "eligible_for_submission",
                "run_id": "ema-learned-division-candidate-v1",
                "target_public_score": 0.945,
                "known_public_hash_match": False,
                "learned_edges_added": 2,
                "learned_reassignments": 0,
                "learned_node_or_coordinate_changes": 0,
                "proxy_gain": 0.01,
                "competition_submission_performed": False,
                "authorized_for_submission": True,
                "submission_path": str(submission),
                "submission_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    module = runpy.run_path(str(SUBMITTER))

    try:
        module["validate_promotion"](promotion)
    except RuntimeError as error:
        assert "promotion evidence is invalid" in str(error)
    else:
        raise AssertionError("mutated submission unexpectedly passed")
