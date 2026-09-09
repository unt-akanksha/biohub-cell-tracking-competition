from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(
    str(ROOT / "scripts/verify-graph-context-fresh-training-v3.py")
)
DATASET_SLUG = "biohub-graph-context-fresh-consensus-v3"
DATASET_ID = f"indarkarhana/{DATASET_SLUG}"
RUN_ID = "competition-graph-context-fresh-deployment-v3"
POLICY_RUN_ID = "competition-graph-context-fresh-policy-v3"
SKLEARN_WHEEL = (
    "scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64."
    "manylinux_2_28_x86_64.whl"
)
RUNTIME_FILES = {
    "learned_division_recovery.py": ROOT / "research/learned_division_recovery.py",
    "graph_context_division_inference.py": ROOT
    / "research/temporal_contrastive/graph_context_division_inference.py",
    "graph_context_division_model.py": ROOT
    / "research/temporal_contrastive/graph_context_division_model.py",
    "graph_context_features.py": ROOT
    / "research/temporal_contrastive/graph_context_features.py",
    "relational_division_inference.py": ROOT
    / "research/temporal_contrastive/relational_division_inference.py",
    "relational_division_model.py": ROOT
    / "research/temporal_contrastive/relational_division_model.py",
    "multiscale_contextual_pair_fusion.py": ROOT
    / "research/temporal_contrastive/multiscale_contextual_pair_fusion.py",
    "contextual_pair_fusion.py": ROOT
    / "research/temporal_contrastive/contextual_pair_fusion.py",
    "pair_fusion.py": ROOT / "research/temporal_contrastive/pair_fusion.py",
    "patch_model.py": ROOT / "research/temporal_contrastive/patch_model.py",
    "transition_context.py": ROOT
    / "research/temporal_contrastive/transition_context.py",
    "handcrafted_division.py": ROOT / "research/train_handcrafted_division_gate.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def unique(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, saw {matches}")
    return matches[0]


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "GRAPH_CONTEXT_FRESH_MANIFEST.json"
    policy_path = root / "graph-context-fresh-policy.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    members = policy.get("graph_context_members", [])
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUN_ID
        and 2 <= len(members) <= 4
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and policy.get("status") == "fresh_audit_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("graph_context_policy")
        == "all-selection-admitted-equal-rank-ensemble"
        and policy.get("graph_context_member_count") == len(members)
        and policy.get("fresh_audit_gate_passed") is True
        and policy.get("selection_policy_frozen_before_audit") is True
        and policy.get("maximum_added_edges_per_movie") == 1
        and float(policy.get("biological_geometry_minimum")) == 3.0
        and policy.get("absolute_threshold_used") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
        and all(
            row.get("parameter_count") == 74_732_308
            and row.get("selection_gate_passed") is True
            and row.get("model_sha256") == sha256_file(root / row["path"])
            for row in members
        )
        and policy.get("morphology_model_sha256")
        == sha256_file(root / "morphology_model.joblib")
    ):
        raise ValueError("fresh deployment policy is ineligible")
    if set(files) != {
        *RUNTIME_FILES,
        *(row["path"] for row in members),
        "fresh_training_terminal.json",
        "fresh_selection_policy.json",
        "fresh_verification_report.json",
        "morphology_model.joblib",
        "graph-context-fresh-policy.json",
        SKLEARN_WHEEL,
    }:
        raise ValueError("fresh deployment file inventory changed")
    for name, record in files.items():
        path = root / name
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"fresh deployment file changed: {name}")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "member_count": len(members),
        "authorized_for_full_candidate_evaluation": True,
        "authorized_for_submission": False,
    }


def build(results_root: Path, wheel: Path, output_root: Path) -> dict[str, Any]:
    report = VERIFY["verify"](results_root)
    if report.get("authorized_for_full_candidate_evaluation") is not True:
        raise ValueError("fresh graph training did not pass audit")
    terminal_path = unique(results_root, "graph_context_fresh_ensemble_terminal.json")
    source_root = terminal_path.parent
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    selection_path = source_root / "selection_policy.json"
    morphology_path = source_root / "morphology_model.joblib"
    if wheel.name != SKLEARN_WHEEL or not wheel.is_file():
        raise ValueError("the pinned CPython 3.12 sklearn wheel is required")
    output_root.mkdir(parents=True, exist_ok=False)
    copies = {
        **RUNTIME_FILES,
        "fresh_training_terminal.json": terminal_path,
        "fresh_selection_policy.json": selection_path,
        "morphology_model.joblib": morphology_path,
        "fresh_verification_report.json": None,
        SKLEARN_WHEEL: wheel,
    }
    member_records = []
    for index, member in enumerate(report["deployment_members"]):
        worker_path = source_root / member / "worker_terminal.json"
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        checkpoint = source_root / member / "graph_context_model.pt"
        name = f"graph_context_model_{index:02d}.pt"
        copies[name] = checkpoint
        member_records.append(
            {
                "path": name,
                "member": member,
                "seed": worker["seed"],
                "model_sha256": worker["model_sha256"],
                "parameter_count": 74_732_308,
                "selection_average_precision": worker["selection"]["average_precision"],
                "selection_gate_passed": True,
            }
        )
    for name, source in copies.items():
        if source is not None:
            shutil.copy2(source, output_root / name)
    write_json(output_root / "fresh_verification_report.json", report)
    policy = {
        "schema_version": 1,
        "status": "fresh_audit_accepted",
        "run_id": POLICY_RUN_ID,
        "source_run_id": terminal["run_id"],
        "source_terminal_sha256": sha256_file(terminal_path),
        "source_selection_policy_sha256": sha256_file(selection_path),
        "graph_context_policy": "all-selection-admitted-equal-rank-ensemble",
        "graph_context_member_count": len(member_records),
        "graph_context_members": member_records,
        "fresh_audit_gate_passed": True,
        "fresh_audit_consensus": terminal["audit_consensus"],
        "selection_policy_frozen_before_audit": True,
        "morphology_model_sha256": sha256_file(morphology_path),
        "sklearn_version": "1.9.0",
        "sklearn_wheel_sha256": sha256_file(wheel),
        "biological_geometry_minimum": 3.0,
        "maximum_added_edges_per_movie": 1,
        "absolute_threshold_used": False,
        "model_weights_searched_on_audit": False,
        "model_subset_searched_on_audit": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_full_candidate_evaluation": True,
        "authorized_for_submission": False,
    }
    policy_path = output_root / "graph-context-fresh-policy.json"
    write_json(policy_path, policy)
    names = [name for name in copies if name != "fresh_verification_report.json"]
    names.extend(("fresh_verification_report.json", policy_path.name))
    manifest_path = output_root / "GRAPH_CONTEXT_FRESH_MANIFEST.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": RUN_ID,
            "competition_test_data_read": False,
            "public_code_copied": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_command_included": False,
            "files": {
                name: {
                    "bytes": (output_root / name).stat().st_size,
                    "sha256": sha256_file(output_root / name),
                }
                for name in names
            },
        },
    )
    write_json(
        output_root / "dataset-metadata.json",
        {
            "title": "Biohub Graph Context Fresh Consensus v3",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    return verify_dataset(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--sklearn-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.results_root, args.sklearn_wheel, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
