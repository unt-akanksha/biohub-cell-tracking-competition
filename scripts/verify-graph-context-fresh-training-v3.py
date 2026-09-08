from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ID = "competition-graph-context-fresh-ensemble-v3"
SPLIT_SHA256 = "8d53a55217be88efc895ae9bf0bf378a3a4acb6c42437836a342d888cc9296da"
RUNTIME_MANIFEST_SHA256 = (
    "78c4e24b1f6c72157e2c2d8f8416897422b8abd2443968b2fa9d4647c178ea5b"
)
PARAMETER_COUNT = 74_732_308
EXPECTED_MEMBERS = {
    "seed-1409101-init-1": (1_409_101, "target_44b6"),
    "seed-1409101-init-2": (1_419_104, "target_6bba"),
    "seed-1509107-init-1": (1_509_107, "target_44b6"),
    "seed-1509107-init-2": (1_519_110, "target_6bba"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected one {name}, saw {matches}")
    return matches[0]


def verify_member(root: Path, member: str) -> dict[str, Any]:
    terminal_path = unique(root / member, "worker_terminal.json")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    checkpoint = terminal_path.parent / "graph_context_model.pt"
    history_path = terminal_path.parent / "selection_history.json"
    expected_seed, expected_fold = EXPECTED_MEMBERS[member]
    selection = terminal.get("selection", {})
    if not (
        terminal.get("run_id") == RUN_ID
        and terminal.get("member") == member
        and terminal.get("seed") == expected_seed
        and terminal.get("warm_start_fold") == expected_fold
        and terminal.get("fresh_split_sha256") == SPLIT_SHA256
        and terminal.get("completed_steps") == 20_000
        and terminal.get("parameter_count") == PARAMETER_COUNT
        and checkpoint.is_file()
        and terminal.get("model_sha256") == sha256_file(checkpoint)
        and history_path.is_file()
        and terminal.get("audit_opened") is False
        and terminal.get("final_probe_opened") is False
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
        and isinstance(selection, dict)
    ):
        raise ValueError(f"member evidence failed: {member}")
    return terminal


def verify(root: Path) -> dict[str, Any]:
    terminal_path = unique(root, "graph_context_fresh_ensemble_terminal.json")
    output_root = terminal_path.parent
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("run_id") == RUN_ID
        and terminal.get("fresh_split_sha256") == SPLIT_SHA256
        and terminal.get("runtime_manifest_sha256") == RUNTIME_MANIFEST_SHA256
        and terminal.get("required_visible_gpu_count") == 2
        and terminal.get("completed_model_count") == 4
        and terminal.get("competition_test_data_read") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and terminal.get("authorized_for_submission") is False
    ):
        raise ValueError("aggregate terminal integrity failed")
    members = {
        member: verify_member(output_root, member) for member in EXPECTED_MEMBERS
    }
    accepted = terminal.get("status") == "accepted_at_fresh_audit"
    if not accepted:
        if terminal.get("authorized_for_full_candidate_evaluation") is not False:
            raise ValueError("rejected run authorizes candidate evaluation")
        return {
            "schema_version": 1,
            "status": "verified_rejection",
            "run_id": RUN_ID,
            "reason": terminal.get("reason", "fresh_audit_gate_failed"),
            "terminal_sha256": sha256_file(terminal_path),
            "completed_model_count": len(members),
            "authorized_for_full_candidate_evaluation": False,
            "authorized_for_submission": False,
        }

    policy_path = output_root / "selection_policy.json"
    morphology_path = output_root / "morphology_model.joblib"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    admitted = terminal.get("selection_accepted_members", [])
    consensus = terminal.get("audit_consensus", {})
    rows = consensus.get("rows", [])
    recalculated = {
        "selected": len(rows),
        "tp": sum(row.get("target") == 1 for row in rows),
        "fp": sum(row.get("target") == 0 for row in rows),
    }
    if not (
        2 <= len(admitted) <= 4
        and set(admitted) <= set(EXPECTED_MEMBERS)
        and all(members[name].get("selection_gate_passed") is True for name in admitted)
        and policy.get("status") == "frozen_before_audit"
        and policy.get("run_id") == RUN_ID
        and policy.get("deployment_members") == admitted
        and terminal.get("selection_policy_sha256") == sha256_file(policy_path)
        and policy.get("morphology_model_sha256") == sha256_file(morphology_path)
        and terminal.get("morphology_model_sha256") == sha256_file(morphology_path)
        and terminal.get("selection_policy_frozen_before_audit") is True
        and terminal.get("audit_opened_after_policy_freeze") is True
        and terminal.get("policy_audit_passed") is True
        and terminal.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("absolute_threshold_used_for_deployment") is False
        and policy.get("maximum_added_edges_per_movie") == 1
        and float(policy.get("biological_geometry_minimum")) == 3.0
        and consensus.get("selected") == recalculated["selected"]
        and consensus.get("tp") == recalculated["tp"] >= 3
        and consensus.get("fp") == recalculated["fp"] <= 1
        and float(consensus.get("jaccard", 0.0)) > 0.0
        and all(consensus.get("by_embryo", {}).get(prefix, {}).get("tp", 0) > 0 for prefix in ("44b6", "6bba"))
    ):
        raise ValueError("accepted aggregate failed the independent gate")
    return {
        "schema_version": 1,
        "status": "verified_fresh_audit_acceptance",
        "run_id": RUN_ID,
        "terminal_sha256": sha256_file(terminal_path),
        "selection_policy_sha256": sha256_file(policy_path),
        "morphology_model_sha256": sha256_file(morphology_path),
        "deployment_members": admitted,
        "model_sha256": [members[name]["model_sha256"] for name in admitted],
        "audit_consensus": consensus,
        "authorized_for_full_candidate_evaluation": True,
        "authorized_for_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.output_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
