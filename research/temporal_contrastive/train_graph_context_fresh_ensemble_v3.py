#!/usr/bin/env python
"""Train or aggregate the clean graph-context fresh-split v3 ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.temporal_contrastive import train_graph_context_division_sweep as base
from research.train_handcrafted_division_gate import patch_features


RUN_ID = "competition-graph-context-fresh-ensemble-v3"
SPLIT_RUN_ID = "competition-graph-context-fresh-split-v3"
SPLIT_SHA256 = "8d53a55217be88efc895ae9bf0bf378a3a4acb6c42437836a342d888cc9296da"
WARM_STARTS = {
    "target_44b6": "a0a1794134893d191d896b8b83abee61a754bc3852e2b8f74dacd353ce118a76",
    "target_6bba": "9e8af9aeb247297d3ed09bda3bcc2b5af413d78c6bf2546eff5f4898667e074d",
}
SEEDS = (1_409_101, 1_509_107)
STEPS = 20_000
BATCH_SIZE = 10
VALIDATION_BATCH_SIZE = 20
VALIDATION_EVERY = 250
LEARNING_RATE = 6e-5
MINIMUM_LEARNING_RATE = 3e-7
BACKBONE_LR_MULTIPLIER = 0.15
WEIGHT_DECAY = 2e-4
EMA_DECAY = 0.995
GRADIENT_CLIP = 2.0


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


def load_contract(split_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if sha256_file(split_path) != SPLIT_SHA256:
        raise ValueError("fresh split artifact changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    experiment = split.get("frozen_experiment", {})
    roles = {
        str(row["stem"]): str(row["role"])
        for row in split.get("stem_roles", [])
    }
    if not (
        split.get("schema_version") == 1
        and split.get("status") == "frozen_before_model_scoring"
        and split.get("run_id") == SPLIT_RUN_ID
        and split.get("model_outputs_used_for_assignment") is False
        and split.get("leaderboard_used_for_assignment") is False
        and split.get("audit_model_predictions_generated") is False
        and split.get("audit_scores_opened") is False
        and experiment.get("parameter_count_per_member") == 74_732_308
        and experiment.get("external_only_warm_starts") == WARM_STARTS
        and experiment.get("seeds") == list(SEEDS)
        and experiment.get("planned_model_count") == 4
        and experiment.get("steps_per_model") == STEPS
        and experiment.get("metric_hack_allowed") is False
        and len(roles) == 146
        and set(roles.values()) == {"optimization", "selection", "audit"}
    ):
        raise ValueError("fresh split contract is ineligible")
    return split, roles


def load_fresh_role(
    data_root: Path,
    source_manifest: dict[str, Any],
    roles: dict[str, str],
    fresh_role: str,
) -> base.GraphContextData:
    """Read only records assigned to one new role, preserving source checks."""

    if fresh_role not in {"optimization", "selection", "audit"}:
        raise ValueError(f"unsupported fresh role: {fresh_role}")
    pieces: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "patches",
            "geometry",
            "context",
            "mask",
            "targets",
            "weights",
            "eligible",
        )
    }
    inventory: list[dict[str, Any]] = []
    records = [
        record
        for record in source_manifest["records"]
        if roles[str(record["stem"])] == fresh_role
    ]
    if not records:
        raise ValueError(f"fresh {fresh_role} inventory is empty")
    for record in records:
        path = base._verified_shard(data_root, record)
        with np.load(path, allow_pickle=False) as data:
            arrays = {
                "patches": np.asarray(data["relational_patches"], dtype=np.float16),
                "geometry": np.asarray(data["geometry_features"], dtype=np.float32),
                "context": np.asarray(data["graph_context_features"], dtype=np.float16),
                "mask": np.asarray(data["graph_context_mask"], dtype=np.bool_),
                "targets": np.asarray(
                    data["division_recovery_target"], dtype=np.float32
                ),
                "weights": np.asarray(data["label_weight"], dtype=np.float32),
                "eligible": np.asarray(
                    data["inference_geometry_eligible"], dtype=np.bool_
                ),
            }
            metadata = json.loads(str(data["metadata_json"].item()))
        rows = len(arrays["targets"])
        if not (
            arrays["patches"].shape == (rows, 3, 3, 17, 17, 17)
            and arrays["geometry"].shape == (rows, 9)
            and arrays["context"].shape
            == (rows, base.CONTEXT_TOKEN_COUNT, base.CONTEXT_FEATURE_WIDTH)
            and arrays["mask"].shape == (rows, base.CONTEXT_TOKEN_COUNT)
            and arrays["targets"].shape
            == arrays["weights"].shape
            == arrays["eligible"].shape
            == (rows,)
            and metadata.get("stem") == record["stem"]
            and metadata.get("role") == record["role"]
            and metadata.get("audit_labels_scored") is False
            and metadata.get("competition_test_data_read") is False
        ):
            raise ValueError(f"fresh-role shard contract changed: {path}")
        for key, value in arrays.items():
            pieces[key].append(torch.from_numpy(value))
        inventory.extend(
            {
                "stem": str(record["stem"]),
                "embryo": str(record["embryo"]),
                "timepoint": int(record["timepoint"]),
                "source_row": index,
                "source_role": str(record["role"]),
                "fresh_role": fresh_role,
            }
            for index in range(rows)
        )
    result = tuple(torch.cat(pieces[key]) for key in pieces)
    if not torch.any(result[4] > 0.5) or not torch.any(result[4] < 0.5):
        raise RuntimeError(f"fresh {fresh_role} inventory lost a class")
    return (*result, inventory)  # type: ignore[return-value]


def load_source(data_root: Path) -> dict[str, Any]:
    manifest_path = data_root / "graph_context_relational_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base.validate_manifest(manifest)
    return manifest


def verify_warm_start(
    model_path: Path, terminal_path: Path, fold: str
) -> dict[str, Any]:
    expected = WARM_STARTS[fold]
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    worker = terminal.get("folds", {}).get(fold, {})
    if not (
        sha256_file(model_path) == expected
        and terminal.get("run_id") == "zebrahub-multiscale-contextual-pretrain-v1"
        and terminal.get("competition_data_read") is False
        and terminal.get("public_code_copied") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and worker.get("model_sha256") == expected
        and worker.get("parameter_count") == 46_386_607
        and worker.get("competition_data_read") is False
    ):
        raise ValueError(f"external warm start is ineligible: {fold}")
    return worker


def frozen_training_args() -> argparse.Namespace:
    return argparse.Namespace(
        steps=STEPS,
        batch_size=BATCH_SIZE,
        validation_batch_size=VALIDATION_BATCH_SIZE,
        validation_every=VALIDATION_EVERY,
        log_every=100,
        learning_rate=LEARNING_RATE,
        minimum_learning_rate=MINIMUM_LEARNING_RATE,
        backbone_lr_multiplier=BACKBONE_LR_MULTIPLIER,
        weight_decay=WEIGHT_DECAY,
        ema_decay=EMA_DECAY,
        gradient_clip=GRADIENT_CLIP,
    )


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each fresh graph worker requires one isolated GPU")
    if args.seed not in SEEDS or args.fold not in WARM_STARTS:
        raise ValueError("worker identity is outside the frozen contract")
    initial_index = 1 if args.fold == "target_44b6" else 2
    expected_member = f"seed-{args.seed}-init-{initial_index}"
    if args.member != expected_member:
        raise ValueError("worker member name changed")
    _, roles = load_contract(args.split)
    source = load_source(args.data_root)
    verify_warm_start(args.initial_model, args.warm_start_terminal, args.fold)
    optimization = load_fresh_role(args.data_root, source, roles, "optimization")
    selection = load_fresh_role(args.data_root, source, roles, "selection")
    base.RUN_ID = RUN_ID
    terminal = base.train_member(
        member_name=args.member,
        seed=args.seed + (0 if args.fold == "target_44b6" else 10_003),
        initial_model_path=args.initial_model,
        train_data=optimization,
        selection_data=selection,
        output_root=args.output_root,
        args=frozen_training_args(),
        device=torch.device("cuda:0"),
    )
    terminal["fresh_split_sha256"] = SPLIT_SHA256
    terminal["warm_start_fold"] = args.fold
    terminal["warm_start_competition_data_read"] = False
    atomic_json(args.output_root / args.member / "worker_terminal.json", terminal)


def member_specs() -> list[tuple[str, int, str]]:
    return [
        (f"seed-{seed}-init-{index}", seed, fold)
        for seed in SEEDS
        for index, fold in enumerate(WARM_STARTS, start=1)
    ]


def load_member_terminals(output_root: Path) -> list[dict[str, Any]]:
    terminals = []
    for member, seed, fold in member_specs():
        member_root = output_root / member
        terminal_path = member_root / "worker_terminal.json"
        checkpoint = member_root / "graph_context_model.pt"
        history_path = member_root / "selection_history.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        expected_seed = seed + (0 if fold == "target_44b6" else 10_003)
        if not (
            terminal.get("run_id") == RUN_ID
            and terminal.get("member") == member
            and terminal.get("seed") == expected_seed
            and terminal.get("completed_steps") == STEPS
            and terminal.get("fresh_split_sha256") == SPLIT_SHA256
            and terminal.get("warm_start_fold") == fold
            and terminal.get("warm_start_competition_data_read") is False
            and terminal.get("initial_backbone_sha256") == WARM_STARTS[fold]
            and terminal.get("model_sha256") == sha256_file(checkpoint)
            and history_path.is_file()
            and terminal.get("audit_opened") is False
            and terminal.get("competition_test_data_read") is False
            and terminal.get("public_predictions_copied") is False
            and terminal.get("public_leaderboard_used_for_selection") is False
        ):
            raise ValueError(f"fresh member failed integrity checks: {member}")
        terminals.append(terminal)
    return terminals


def predict_members(
    output_root: Path,
    terminals: list[dict[str, Any]],
    data: base.GraphContextData,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    scores = {}
    for terminal in terminals:
        member = str(terminal["member"])
        model = base.GraphContextDivisionModel().to(device)
        checkpoint = output_root / member / "graph_context_model.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True), strict=True
        )
        scores[member] = base.predict(
            model, data, batch_size=VALIDATION_BATCH_SIZE, device=device
        )
        del model
        torch.cuda.empty_cache()
    return scores


def fit_morphology(
    optimization: base.GraphContextData,
    selection: base.GraphContextData,
    output_path: Path,
) -> tuple[dict[str, Any], torch.Tensor]:
    train_x = patch_features(optimization[0][:, 0].float().numpy())
    train_y = optimization[4].numpy().astype(np.int64)
    train_weight = optimization[5].numpy().astype(np.float64)
    selection_x = patch_features(selection[0][:, 0].float().numpy())
    balance = float(
        train_weight[train_y == 0].sum() / max(train_weight[train_y == 1].sum(), 1e-8)
    )
    weights = train_weight * np.where(train_y == 1, balance, 1.0)
    models = [
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000, random_state=2209109),
        ),
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features=0.75,
            n_jobs=-1,
            random_state=2209109,
        ),
    ]
    models[0].fit(
        train_x,
        train_y,
        logisticregression__sample_weight=weights,
    )
    models[1].fit(train_x, train_y, sample_weight=weights)
    joblib.dump(
        {
            "run_id": RUN_ID,
            "feature_family": "temporal_radial_peak_morphology_v1",
            "feature_count": 132,
            "models": models,
        },
        output_path,
        compress=3,
    )
    probabilities = np.mean(
        np.stack([model.predict_proba(selection_x)[:, 1] for model in models]), axis=0
    )
    scores = torch.from_numpy(probabilities.astype(np.float32))
    metrics = base.eligible_metrics(selection[4], scores, selection[6], selection[7])
    return metrics, scores


def score_morphology(model_path: Path, data: base.GraphContextData) -> torch.Tensor:
    payload = joblib.load(model_path)
    if not (
        payload.get("run_id") == RUN_ID
        and payload.get("feature_family") == "temporal_radial_peak_morphology_v1"
        and payload.get("feature_count") == 132
        and len(payload.get("models", [])) == 2
    ):
        raise ValueError("fresh morphology model changed")
    features = patch_features(data[0][:, 0].float().numpy())
    probabilities = np.mean(
        np.stack(
            [model.predict_proba(features)[:, 1] for model in payload["models"]]
        ),
        axis=0,
    )
    return torch.from_numpy(probabilities.astype(np.float32))


def consensus_audit(
    targets: torch.Tensor,
    eligible: torch.Tensor,
    graph_scores: torch.Tensor,
    morphology_scores: torch.Tensor,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(inventory):
        if bool(eligible[index]):
            groups.setdefault(str(row["stem"]), []).append(index)
    selected = []
    for stem, indices in sorted(groups.items()):
        graph_top = indices[int(torch.argmax(graph_scores[indices]))]
        morphology_top = indices[int(torch.argmax(morphology_scores[indices]))]
        if graph_top != morphology_top:
            continue
        index = graph_top
        selected.append(
            {
                "stem": stem,
                "embryo": str(inventory[index]["embryo"]),
                "timepoint": int(inventory[index]["timepoint"]),
                "source_row": int(inventory[index]["source_row"]),
                "target": int(targets[index] > 0.5),
                "graph_score": float(graph_scores[index]),
                "morphology_score": float(morphology_scores[index]),
            }
        )
    tp = sum(row["target"] == 1 for row in selected)
    fp = sum(row["target"] == 0 for row in selected)
    positives = int(((targets > 0.5) & eligible).sum())
    fn = positives - tp
    by_embryo = {
        embryo: {
            "selected": sum(row["embryo"] == embryo for row in selected),
            "tp": sum(
                row["embryo"] == embryo and row["target"] == 1
                for row in selected
            ),
            "fp": sum(
                row["embryo"] == embryo and row["target"] == 0
                for row in selected
            ),
        }
        for embryo in ("44b6", "6bba")
    }
    return {
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "jaccard": tp / max(tp + fp + fn, 1),
        "by_embryo": by_embryo,
        "rows": selected,
    }


def rejection_terminal(
    output_root: Path,
    *,
    reason: str,
    terminals: list[dict[str, Any]],
    selection_ensemble: dict[str, Any] | None = None,
    morphology_selection: dict[str, Any] | None = None,
) -> None:
    atomic_json(
        output_root / "graph_context_fresh_ensemble_terminal.json",
        {
            "schema_version": 1,
            "status": "rejected_at_selection",
            "run_id": RUN_ID,
            "reason": reason,
            "fresh_split_sha256": SPLIT_SHA256,
            "completed_model_count": len(terminals),
            "selection_accepted_members": [
                row["member"] for row in terminals if row["selection_gate_passed"]
            ],
            "selection_ensemble": selection_ensemble,
            "morphology_selection": morphology_selection,
            "selection_policy_frozen": False,
            "audit_opened": False,
            "competition_test_data_read": False,
            "public_predictions_copied": False,
            "public_leaderboard_used_for_selection": False,
            "submission_created": False,
            "authorized_for_full_candidate_evaluation": False,
            "authorized_for_submission": False,
        },
    )


def run_aggregate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fresh aggregation requires one isolated GPU")
    started = time.monotonic()
    split, roles = load_contract(args.split)
    source = load_source(args.data_root)
    terminals = load_member_terminals(args.output_root)
    selection = load_fresh_role(args.data_root, source, roles, "selection")
    accepted = [row for row in terminals if row["selection_gate_passed"]]
    if len(accepted) < 2:
        rejection_terminal(
            args.output_root,
            reason="fewer_than_two_selection_admitted_members",
            terminals=terminals,
        )
        raise SystemExit(2)
    device = torch.device("cuda:0")
    selection_scores = predict_members(args.output_root, accepted, selection, device)
    selection_ensemble_scores = base.calibration_free_equal_rank_ensemble(
        [selection_scores[row["member"]] for row in accepted]
    )
    selection_ensemble = base.eligible_metrics(
        selection[4], selection_ensemble_scores, selection[6], selection[7]
    )
    if not base.passes_selection_gate(selection_ensemble):
        rejection_terminal(
            args.output_root,
            reason="equal_rank_selection_ensemble_failed",
            terminals=terminals,
            selection_ensemble=selection_ensemble,
        )
        raise SystemExit(2)

    optimization = load_fresh_role(args.data_root, source, roles, "optimization")
    morphology_path = args.output_root / "morphology_model.joblib"
    morphology_selection, _ = fit_morphology(
        optimization, selection, morphology_path
    )
    if not base.passes_selection_gate(morphology_selection):
        rejection_terminal(
            args.output_root,
            reason="morphology_selection_gate_failed",
            terminals=terminals,
            selection_ensemble=selection_ensemble,
            morphology_selection=morphology_selection,
        )
        raise SystemExit(2)

    selected_members = [str(row["member"]) for row in accepted]
    selection_policy = {
        "schema_version": 1,
        "status": "frozen_before_audit",
        "run_id": RUN_ID,
        "fresh_split_sha256": SPLIT_SHA256,
        "deployment_policy": "all-selection-admitted-equal-rank-ensemble",
        "deployment_members": selected_members,
        "member_sha256": {
            member: sha256_file(
                args.output_root / member / "graph_context_model.pt"
            )
            for member in selected_members
        },
        "selection_ensemble": selection_ensemble,
        "morphology_selection": morphology_selection,
        "morphology_model_sha256": sha256_file(morphology_path),
        "morphology_estimators": ["logistic_c0.1", "extra_trees_leaf5"],
        "absolute_threshold_used_for_deployment": False,
        "maximum_added_edges_per_movie": 1,
        "biological_geometry_minimum": 3.0,
        "audit_opened": False,
        "competition_test_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "authorized_for_submission": False,
    }
    policy_path = args.output_root / "selection_policy.json"
    atomic_json(policy_path, selection_policy)

    # This is the first point at which any new-audit shard is opened.
    audit = load_fresh_role(args.data_root, source, roles, "audit")
    audit_scores = predict_members(args.output_root, accepted, audit, device)
    audit_ensemble_scores = base.calibration_free_equal_rank_ensemble(
        [audit_scores[member] for member in selected_members]
    )
    audit_ensemble = base.eligible_metrics(
        audit[4], audit_ensemble_scores, audit[6], audit[7]
    )
    morphology_audit_scores = score_morphology(morphology_path, audit)
    morphology_audit = base.eligible_metrics(
        audit[4], morphology_audit_scores, audit[6], audit[7]
    )
    consensus = consensus_audit(
        audit[4], audit[6], audit_ensemble_scores, morphology_audit_scores, audit[7]
    )
    audit_passed = bool(
        consensus["tp"] >= 3
        and consensus["fp"] <= 1
        and consensus["jaccard"] > 0.0
        and all(
            consensus["by_embryo"][embryo]["tp"] > 0
            for embryo in ("44b6", "6bba")
        )
    )
    terminal = {
        "schema_version": 1,
        "status": (
            "accepted_at_fresh_audit" if audit_passed else "rejected_at_fresh_audit"
        ),
        "run_id": RUN_ID,
        "elapsed_seconds": time.monotonic() - started,
        "fresh_split_sha256": SPLIT_SHA256,
        "fresh_split_source_manifest_sha256": split["source_manifest_sha256"],
        "planned_model_count": 4,
        "completed_model_count": len(terminals),
        "selection_accepted_members": selected_members,
        "selection_policy_sha256": sha256_file(policy_path),
        "selection_policy_frozen_before_audit": True,
        "selection_ensemble": selection_ensemble,
        "morphology_selection": morphology_selection,
        "audit_ensemble": audit_ensemble,
        "morphology_audit": morphology_audit,
        "audit_consensus": consensus,
        "audit_opened": True,
        "audit_opened_after_policy_freeze": True,
        "policy_audit_passed": audit_passed,
        "deployment_policy": (
            "graph-context/morphology top-rank agreement" if audit_passed else None
        ),
        "deployment_members": selected_members if audit_passed else [],
        "morphology_model_sha256": sha256_file(morphology_path),
        "absolute_threshold_used_for_deployment": False,
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_full_candidate_evaluation": audit_passed,
        "authorized_for_submission": False,
    }
    atomic_json(
        args.output_root / "graph_context_fresh_ensemble_terminal.json", terminal
    )
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    if not audit_passed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--data-root", type=Path, required=True)
    worker.add_argument("--split", type=Path, required=True)
    worker.add_argument("--initial-model", type=Path, required=True)
    worker.add_argument("--warm-start-terminal", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--member", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--fold", choices=tuple(WARM_STARTS), required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--data-root", type=Path, required=True)
    aggregate.add_argument("--split", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
