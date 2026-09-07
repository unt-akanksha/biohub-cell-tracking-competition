#!/usr/bin/env python
"""Verify and promote an additive graph-context-consensus Kaggle output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_BASE = runpy.run_path(
    str(ROOT / "scripts/verify-relational-consensus-submission-candidate.py")
)
sha256_file = _BASE["sha256_file"]
RUN_ID = "ema-graph-context-consensus-candidate-v1"
RUNTIME_RUN_ID = "competition-graph-context-consensus-division-dataset-v1"
POLICY_RUN_ID = "competition-graph-context-consensus-division-policy-v1"
EXPECTED_PARAMETER_COUNT = 74_732_308
MINIMUM_PROXY_GAIN = 0.005
MAXIMUM_ADJUSTED_EDGE_REGRESSION = 0.001
V2_POLICY_CONTRACT = "all-selection-admitted-equal-rank-ensemble-v2"
PUBLIC_CONTROL_VALIDATOR_SHA256 = _BASE["PUBLIC_CONTROL_VALIDATOR_SHA256"]


def validate_runtime(runtime_manifest: Path) -> dict[str, Any]:
    root = runtime_manifest.parent
    manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    policy_path = root / "graph-context-consensus-policy.json"
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("status") == "complete"
        and manifest.get("run_id") == RUNTIME_RUN_ID
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_code_copied") is False
        and manifest.get("public_predictions_copied") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
        and manifest.get("submission_command_included") is False
        and "graph-context-consensus-policy.json" in files
        and policy_path.is_file()
    ):
        raise RuntimeError("graph-context runtime manifest is invalid")
    for name, record in files.items():
        path = root / name
        if not (
            path.is_file()
            and record.get("path") == name
            and record.get("sha256") == sha256_file(path)
        ):
            raise RuntimeError(f"graph-context runtime file changed: {name}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    members = policy.get("graph_context_members", [])
    hashes = [row.get("model_sha256") for row in members]
    constituent_audit_contract = bool(
        (
            policy.get("constituent_audit_gate_required", True) is True
            and all(row.get("audit_gate_passed") is True for row in members)
        )
        or (
            policy.get("constituent_audit_gate_required") is False
            and policy.get("policy_unit_audited") is True
            and policy.get("policy_contract") == V2_POLICY_CONTRACT
            and policy.get("graph_context_policy")
            == "equal_rank_selection_admitted_ensemble"
            and len(members) >= 2
            and all(isinstance(row.get("audit_gate_passed"), bool) for row in members)
        )
    )
    if not (
        policy.get("schema_version") == 1
        and policy.get("status") == "development_accepted"
        and policy.get("run_id") == POLICY_RUN_ID
        and policy.get("graph_context_policy")
        in {"equal_rank_selection_admitted_ensemble", "strongest_selection_individual"}
        and 1 <= len(members) <= 8
        and policy.get("graph_context_member_count") == len(members)
        and len(set(hashes)) == len(hashes)
        and all(
            row.get("path") in files
            and row.get("model_sha256") == sha256_file(root / row["path"])
            and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("selection_gate_passed") is True
            for row in members
        )
        and constituent_audit_contract
        and policy.get("base_safe_division_heuristic_enabled") is True
        and policy.get("external_policy_additive_only") is True
        and policy.get("maximum_added_edges_per_movie") == 1
        and policy.get("exact_two_t4_required") is True
        and policy.get("absolute_threshold_used") is False
        and policy.get("model_subset_searched_on_audit") is False
        and policy.get("authorized_for_full_candidate_evaluation") is True
        and policy.get("authorized_for_submission") is False
    ):
        raise RuntimeError("graph-context runtime policy is invalid")
    return {
        "manifest_sha256": sha256_file(runtime_manifest),
        "policy_sha256": sha256_file(policy_path),
        "deep_policy": policy["graph_context_policy"],
        "deep_member_count": len(members),
        "deep_model_sha256": hashes,
        "morphology_model_sha256": policy["morphology_model_sha256"],
        "expected_gpu_groups": 2 if len(members) >= 2 else 1,
    }


_GRAPH_GLOBALS = _BASE["verify_candidate"].__globals__
_GRAPH_GLOBALS.update(
    {
        "RUN_ID": RUN_ID,
        "RUNTIME_RUN_ID": RUNTIME_RUN_ID,
        "POLICY_RUN_ID": POLICY_RUN_ID,
        "EXPECTED_PARAMETER_COUNT": EXPECTED_PARAMETER_COUNT,
        "MINIMUM_PROXY_GAIN": MINIMUM_PROXY_GAIN,
        "MAXIMUM_ADJUSTED_EDGE_REGRESSION": MAXIMUM_ADJUSTED_EDGE_REGRESSION,
        "validate_runtime": validate_runtime,
    }
)
verify_candidate = _BASE["verify_candidate"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-validator", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify_candidate(
        args.output_root,
        args.baseline_validator,
        args.runtime_manifest,
        expected_baseline_sha256=PUBLIC_CONTROL_VALIDATOR_SHA256,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")


if __name__ == "__main__":
    main()
