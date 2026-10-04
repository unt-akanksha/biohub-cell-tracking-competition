from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V3_DEPLOY = runpy.run_path(
    str(ROOT / "scripts/build-graph-context-fresh-deployment-v3.py"),
    run_name="unanimous_salvage_v4_base_deployment",
)
V3_VERIFY = runpy.run_path(
    str(ROOT / "scripts/verify-graph-context-fresh-training-v3.py"),
    run_name="unanimous_salvage_v4_source_verifier",
)
TRAIN = runpy.run_path(
    str(ROOT / "research/temporal_contrastive/train_graph_context_fresh_ensemble_v3.py"),
    run_name="unanimous_salvage_v4_training_contract",
)
DATASET_SLUG = "biohub-graph-context-unanimous-salvage-v4"
DATASET_ID = f"indarkarhana/{DATASET_SLUG}"
RUN_ID = "competition-graph-context-unanimous-salvage-deployment-v4"
POLICY_RUN_ID = "competition-graph-context-unanimous-salvage-v4"
V3_TERMINAL_SHA256 = (
    "92f1150d5579cdbec41c92e3a04525a67d284975d911e42fb1ed29c54c8cf5dd"
)
POLICY_SHA256 = (
    "f8c2ea4e8c8ec2064b19cf1ce868c8a952694053e612a156eee3acddf2b42e6d"
)
SKLEARN_WHEEL = V3_DEPLOY["SKLEARN_WHEEL"]
RUNTIME_FILES = V3_DEPLOY["RUNTIME_FILES"]
sha256_file = V3_DEPLOY["sha256_file"]
write_json = V3_DEPLOY["write_json"]


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "GRAPH_CONTEXT_SALVAGE_MANIFEST.json"
    policy_path = root / "graph-context-unanimous-salvage-policy.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    members = policy.get("members", [])
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "evaluation_only"
        and manifest.get("run_id") == RUN_ID
        and manifest.get("competition_test_data_read") is False
        and manifest.get("submission_command_included") is False
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("status") == "frozen_before_cross_family_movie_evaluation"
        and policy.get("source_v3_terminal_sha256") == V3_TERMINAL_SHA256
        and policy.get("minimum_member_agreement") == 4
        and policy.get("morphology_top_parent_agreement_required") is True
        and policy.get("maximum_added_edges_per_movie") == 1
        and policy.get("cross_family_stems_seen_by_graph_training") is False
        and policy.get("audit_scores_used_to_set_member_thresholds") is False
        and policy.get("leaderboard_used_for_policy_selection") is False
        and policy.get("metric_hack_used") is False
        and policy.get("authorized_for_submission") is False
        and len(members) == 4
        and all(
            row.get("parameter_count") == 74_732_308
            and row.get("selection_threshold_evidence", {}).get("fp") == 0
            and row.get("selection_threshold_evidence", {}).get("tp", 0) >= 2
            and row.get("raw_logit_threshold")
            == row.get("selection_threshold_evidence", {}).get("threshold")
            and row.get("model_sha256") == sha256_file(root / row["path"])
            for row in members
        )
    ):
        raise ValueError("unanimous salvage deployment policy is ineligible")
    expected = {
        *RUNTIME_FILES,
        *(row["path"] for row in members),
        "graph-context-unanimous-salvage-policy.json",
        "v3_training_terminal.json",
        "v3_selection_policy.json",
        "morphology_model.joblib",
        SKLEARN_WHEEL,
    }
    if set(files) != expected:
        raise ValueError("unanimous salvage deployment inventory changed")
    for name, record in files.items():
        path = root / name
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"unanimous salvage deployment file changed: {name}")
    return {
        "status": "verified_evaluation_only",
        "manifest_sha256": sha256_file(manifest_path),
        "member_count": len(members),
        "authorized_for_full_candidate_evaluation": True,
        "authorized_for_submission": False,
    }


def build(
    results_root: Path,
    policy_path: Path,
    wheel_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    verification = V3_VERIFY["verify"](results_root)
    if not (
        verification.get("status") == "verified_rejection"
        and verification.get("reason") == "fresh_audit_gate_failed"
        and verification.get("terminal_sha256") == V3_TERMINAL_SHA256
    ):
        raise ValueError("v3 source is not the pinned rejected training run")
    if sha256_file(policy_path) != POLICY_SHA256:
        raise ValueError("unanimous salvage policy policy changed")
    if wheel_path.name != SKLEARN_WHEEL or not wheel_path.is_file():
        raise ValueError("pinned sklearn wheel is unavailable")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    terminals = TRAIN["load_member_terminals"](results_root)
    terminal_by_member = {row["member"]: row for row in terminals}
    if set(terminal_by_member) != {row["member"] for row in policy["members"]}:
        raise ValueError("unanimous policy member inventory changed")
    output_root.mkdir(parents=True, exist_ok=False)

    deployed_members = []
    for index, row in enumerate(policy["members"]):
        member = str(row["member"])
        source = results_root / member / "graph_context_model.pt"
        name = f"graph_context_model_{index:02d}.pt"
        if sha256_file(source) != row["model_sha256"]:
            raise ValueError(f"unanimous salvage member changed: {member}")
        shutil.copy2(source, output_root / name)
        deployed_members.append({**row, "path": name})

    for name, source in RUNTIME_FILES.items():
        shutil.copy2(source, output_root / name)
    shutil.copy2(
        results_root / "graph_context_fresh_ensemble_terminal.json",
        output_root / "v3_training_terminal.json",
    )
    shutil.copy2(
        results_root / "selection_policy.json",
        output_root / "v3_selection_policy.json",
    )
    shutil.copy2(
        results_root / "morphology_model.joblib",
        output_root / "morphology_model.joblib",
    )
    shutil.copy2(wheel_path, output_root / SKLEARN_WHEEL)
    deployed_policy = {
        **policy,
        "members": deployed_members,
        "source_policy_sha256": POLICY_SHA256,
    }
    deployed_policy_path = (
        output_root / "graph-context-unanimous-salvage-policy.json"
    )
    write_json(deployed_policy_path, deployed_policy)

    names = {
        *RUNTIME_FILES,
        *(row["path"] for row in deployed_members),
        "graph-context-unanimous-salvage-policy.json",
        "v3_training_terminal.json",
        "v3_selection_policy.json",
        "morphology_model.joblib",
        SKLEARN_WHEEL,
    }
    write_json(
        output_root / "GRAPH_CONTEXT_SALVAGE_MANIFEST.json",
        {
            "schema_version": 1,
            "status": "evaluation_only",
            "run_id": RUN_ID,
            "competition_test_data_read": False,
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
            "title": "Biohub Graph Context Unanimous Salvage v4",
            "id": DATASET_ID,
            "licenses": [{"name": "MIT"}],
            "isPrivate": True,
        },
    )
    return verify_dataset(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--sklearn-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.results_root,
        args.policy,
        args.sklearn_wheel,
        args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
