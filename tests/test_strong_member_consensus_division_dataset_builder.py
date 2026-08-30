from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build-strong-member-consensus-division-dataset.py"
MODULE = runpy.run_path(str(SCRIPT))
POLICY = runpy.run_path(
    str(ROOT / "research/temporal_contrastive/overnight_seed_policy.py")
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_sources(tmp_path: Path) -> dict[str, Path]:
    seed = 205_043
    fold = "target_6bba"
    sweep = tmp_path / "sweep"
    model = sweep / f"seed-{seed}" / fold / "division_model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"independently-trained-large-model")
    model_hash = sha256(model)
    selection_ap = POLICY["REFERENCE_SELECTION_AP"] + 0.02
    candidate = {
        "seed": seed,
        "fold": fold,
        "status": "admitted",
        "model_sha256": model_hash,
        "selection": {"average_precision": selection_ap},
        "selection_by_embryo": {
            "44b6": {"average_precision": 0.51},
            "6bba": {"average_precision": 0.52},
        },
        "selection_frozen_threshold_diagnostic": {"tp": 3, "fp": 0},
        "parameter_count": 46_386_607,
        "trainable_parameters": 25_178_047,
    }
    selection = tmp_path / "seed_ensemble_terminal.json"
    write_json(
        selection,
        {
            "schema_version": 1,
            "run_id": POLICY["ENSEMBLE_RUN_ID"],
            "status": "stronger_individuals_eligible_for_development_probe",
            "authorized_for_development_probe": True,
            "competition_test_data_read": False,
            "final_probe_opened": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_submission": False,
            "ensemble_eligible_for_development_probe": False,
            "ensemble": None,
            "candidates": [candidate],
            "stronger_individuals": [
                {
                    "seed": seed,
                    "fold": fold,
                    "model_sha256": model_hash,
                    "selection_average_precision": selection_ap,
                }
            ],
        },
    )
    probe = tmp_path / "seed_ensemble_probe.json"
    write_json(
        probe,
        {
            "schema_version": 1,
            "status": "development_probe_complete",
            "run_id": MODULE["PROBE_RUN_ID"],
            "selection_terminal_sha256": sha256(selection),
            "member_policy_source_sha256": sha256(
                ROOT / "research/temporal_contrastive/overnight_seed_policy.py"
            ),
            "selection_policy": "strongest_individual_rank",
            "member_count": 1,
            "members": [
                {
                    "seed": seed,
                    "fold": fold,
                    "model_sha256": model_hash,
                    "selection_average_precision": selection_ap,
                }
            ],
            "absolute_threshold_used": False,
            "weights_searched_on_probe": False,
            "model_subset_searched_on_probe": False,
            "competition_test_data_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_ranked_consensus_development_evaluation": True,
            "authorized_for_submission": False,
        },
    )
    morphology = tmp_path / "morphology"
    morphology.mkdir()
    morph_model = morphology / "handcrafted_division_gate.joblib"
    morph_model.write_bytes(b"independent-morphology-model")
    write_json(
        morphology / "handcrafted_division_gate_terminal.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": MODULE["MORPHOLOGY_RUN_ID"],
            "feature_count": 132,
            "model_sha256": sha256(morph_model),
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
        },
    )
    development = tmp_path / "development.json"
    write_json(
        development,
        {
            "schema_version": 1,
            "status": "development_positive",
            "run_id": MODULE["DEVELOPMENT_RUN_ID"],
            "deep_probe_sha256": sha256(probe),
            "selected": 1,
            "tp": 1,
            "fp": 0,
            "absolute_threshold_used": False,
            "competition_test_data_read": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_full_candidate_evaluation": True,
            "authorized_for_submission": False,
            "pooled": {
                "edge_before": {"jaccard": 0.91},
                "edge_after": {"jaccard": 0.92},
                "division_before": {"tp": 0, "fp": 0},
                "division_after": {"tp": 1, "fp": 0},
            },
        },
    )
    wheel = tmp_path / MODULE["SKLEARN_WHEEL_NAME"]
    wheel.write_bytes(b"pinned-wheel")
    return {
        "sweep": sweep,
        "selection": selection,
        "probe": probe,
        "morphology": morphology,
        "development": development,
        "wheel": wheel,
    }


def run_builder(tmp_path: Path) -> Path:
    sources = build_sources(tmp_path)
    output = tmp_path / "runtime"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sweep-root",
            str(sources["sweep"]),
            "--ensemble-terminal",
            str(sources["selection"]),
            "--probe-output",
            str(sources["probe"]),
            "--morphology-root",
            str(sources["morphology"]),
            "--development-evidence",
            str(sources["development"]),
            "--sklearn-wheel",
            str(sources["wheel"]),
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def test_builder_binds_one_precommitted_policy_and_preserves_base_divisions(
    tmp_path: Path,
) -> None:
    output = run_builder(tmp_path)

    verified = MODULE["verify_dataset"](output)
    policy = json.loads(
        (output / "strong-member-consensus-policy.json").read_text()
    )
    assert verified["deep_member_count"] == 1
    assert policy["base_safe_division_heuristic_enabled"] is True
    assert policy["external_policy_additive_only"] is True
    assert policy["deep_policy"] == "strongest_individual_rank"
    assert sha256(output / policy["deep_members"][0]["path"]) == policy[
        "deep_members"
    ][0]["model_sha256"]


def test_verifier_rejects_disabling_the_clean_base_division_rule(
    tmp_path: Path,
) -> None:
    output = run_builder(tmp_path)
    policy_path = output / "strong-member-consensus-policy.json"
    policy = json.loads(policy_path.read_text())
    policy["base_safe_division_heuristic_enabled"] = False
    write_json(policy_path, policy)
    manifest_path = output / "STRONG_MEMBER_CONSENSUS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][policy_path.name]["sha256"] = sha256(policy_path)
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="policy changed"):
        MODULE["verify_dataset"](output)
