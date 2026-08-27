#!/usr/bin/env python
"""Evaluate frozen v3 checkpoints once on untouched ZSNS001 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        RECIPROCAL_PARENT_LOSS_WEIGHT,
        ContextualPairFusionAssociationModel,
    )
    from submission_sharding import (
        terminate_and_reap_processes,
        visible_cuda_tokens,
    )
    from train_zebrahub_contextual_pretrain import (
        VALIDATION_PARTITION_POLICY,
        discover_shards,
        inventory_sha256,
        preload_shards,
        validate,
        validation_improvement_gate,
    )
    from verify_zebrahub_contextual_acceptance import verify_acceptance
except ModuleNotFoundError:
    from research.submission_sharding import (
        terminate_and_reap_processes,
        visible_cuda_tokens,
    )
    from research.temporal_contrastive.contextual_pair_fusion import (
        CONTEXTUAL_PAIR_FUSION_FAMILY,
        EXPECTED_PARAMETER_COUNT,
        RECIPROCAL_PARENT_LOSS_WEIGHT,
        ContextualPairFusionAssociationModel,
    )
    from research.temporal_contrastive.train_zebrahub_contextual_pretrain import (
        VALIDATION_PARTITION_POLICY,
        discover_shards,
        inventory_sha256,
        preload_shards,
        validate,
        validation_improvement_gate,
    )
    from research.temporal_contrastive.verify_zebrahub_contextual_acceptance import (
        verify_acceptance,
    )


RUN_ID = "zebrahub-contextual-acceptance-evaluation-v1"
PRETRAINING_RUN_ID = "zebrahub-contextual-pretrain-v1"
FOLDS = ("target_44b6", "target_6bba")
FOLD_SEEDS = {"target_44b6": 51_004, "target_6bba": 61_007}
ACCEPTANCE_SOURCE = "ZSNS001"
ACCEPTANCE_ROLE = "external_acceptance"
ACCEPTANCE_SHARDS = 16
EXPECTED_PRETRAINING_DATASET_MANIFEST_SHA256 = (
    "b35738f215413f1ece403ba5c0601adea82e2540c65f37e6465de0d0755cb7bf"
)
EXPECTED_ACCEPTANCE_MANIFEST_SHA256 = (
    "cbbf670dde160e5a927ed84bb9e2a7313abe4506f4798f6afa00680fc8e7c6d0"
)
EXPECTED_ACCEPTANCE_INVENTORY_SHA256 = (
    "e32bc686e14222e43acb8d6247351e286eae8ed6fdb1f4ab5087e55fb0c79667"
)
EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256 = (
    "05ad8b3195aa2786ffb8a2ffcb247d118d1e5cf96026264e0534c468979e6e0e"
)
DEFAULT_HARD_STOP_SECONDS = 3_600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def seed_initialization(seed: int, device: torch.device) -> torch.Generator:
    """Reproduce the exact RNG setup used immediately before v3 construction."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.default_rng(seed)
    augmentation_generator = torch.Generator(device=device)
    augmentation_generator.manual_seed(seed + 77_777)
    return augmentation_generator


def verify_pretraining_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    terminal_path = root / "pretraining_terminal.json"
    if not terminal_path.is_file():
        raise FileNotFoundError("pretraining aggregate terminal is missing")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    folds = terminal.get("folds")
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("status") == "completed"
        and terminal.get("run_id") == PRETRAINING_RUN_ID
        and terminal.get("gpu_count") == 2
        and terminal.get("both_folds_improved") is True
        and terminal.get("competition_data_read") is False
        and terminal.get("public_predictions_copied") is False
        and terminal.get("public_leaderboard_used_for_selection") is False
        and terminal.get("submission_created") is False
        and isinstance(folds, dict)
        and set(folds) == set(FOLDS)
    ):
        raise ValueError("pretraining aggregate is not eligible for acceptance")
    verified_folds: dict[str, Any] = {}
    for fold in FOLDS:
        row = folds[fold]
        fold_root = root / fold
        worker_path = fold_root / "worker_terminal.json"
        config_path = fold_root / "training_config.json"
        model_path = fold_root / "pretrained_model.pt"
        if not all(path.is_file() for path in (worker_path, config_path, model_path)):
            raise FileNotFoundError(f"pretraining fold artifact is missing: {fold}")
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_hash = sha256_file(model_path)
        if worker != row:
            raise ValueError(f"pretraining worker/aggregate divergence: {fold}")
        if not (
            row.get("status") == "completed"
            and row.get("run_id") == PRETRAINING_RUN_ID
            and row.get("appearance_family") == CONTEXTUAL_PAIR_FUSION_FAMILY
            and row.get("fold") == fold
            and row.get("selection_gate_passed") is True
            and row.get("audit_gate_passed") is True
            and int(row.get("best_step", 0)) > 0
            and row.get("validation_partition_policy") == VALIDATION_PARTITION_POLICY
            and row.get("dataset_manifest_sha256")
            == EXPECTED_PRETRAINING_DATASET_MANIFEST_SHA256
            and row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("reciprocal_parent_loss_weight")
            == RECIPROCAL_PARENT_LOSS_WEIGHT
            and row.get("competition_data_read") is False
            and row.get("public_predictions_copied") is False
            and row.get("public_leaderboard_used_for_selection") is False
            and row.get("submission_created") is False
            and row.get("model_sha256") == model_hash
            and config.get("seed") == FOLD_SEEDS[fold]
            and config.get("fold") == fold
            and config.get("dataset_manifest_sha256")
            == EXPECTED_PRETRAINING_DATASET_MANIFEST_SHA256
        ):
            raise ValueError(f"pretraining fold is not eligible for acceptance: {fold}")
        verified_folds[fold] = {
            "seed": FOLD_SEEDS[fold],
            "model_path": model_path,
            "model_sha256": model_hash,
            "worker_terminal_sha256": sha256_file(worker_path),
            "training_config_sha256": sha256_file(config_path),
            "best_step": int(row["best_step"]),
        }
    return {
        "root": root,
        "terminal_sha256": sha256_file(terminal_path),
        "folds": verified_folds,
    }


def verify_acceptance_source(root: Path) -> tuple[dict[str, Any], list[Any]]:
    evidence = verify_acceptance(root)
    if not (
        evidence.get("status") == "verified_unopened"
        and evidence.get("manifest_sha256") == EXPECTED_ACCEPTANCE_MANIFEST_SHA256
        and evidence.get("inventory_sha256") == EXPECTED_ACCEPTANCE_INVENTORY_SHA256
        and evidence.get("shards") == ACCEPTANCE_SHARDS
        and evidence.get("model_predictions_read") is False
    ):
        raise ValueError("ZSNS001 acceptance source is not frozen and unopened")
    records = discover_shards(
        root / "acceptance",
        expected_source=ACCEPTANCE_SOURCE,
        expected_role=ACCEPTANCE_ROLE,
    )
    if len(records) != ACCEPTANCE_SHARDS:
        raise ValueError("ZSNS001 acceptance shard coverage changed")
    if (
        inventory_sha256(records)
        != EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256
    ):
        raise ValueError("ZSNS001 acceptance inventory changed")
    return evidence, records


def evaluate_worker(args: argparse.Namespace) -> None:
    if args.fold not in FOLDS:
        raise ValueError(f"unknown acceptance fold: {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated acceptance worker requires one GPU, saw {torch.cuda.device_count()}"
        )
    terminal_path = args.output_dir / args.fold / "acceptance_terminal.json"
    if terminal_path.exists():
        raise RuntimeError(f"one-shot acceptance was already opened: {args.fold}")
    started = time.monotonic()
    device = torch.device("cuda:0")
    pretraining = verify_pretraining_source(args.pretraining_root)
    acceptance_evidence, records = verify_acceptance_source(args.acceptance_root)
    seed = FOLD_SEEDS[args.fold]
    seed_initialization(seed, device)
    cache, cache_bytes = preload_shards(records, device)

    initial_model = ContextualPairFusionAssociationModel().to(device)
    if (
        sum(parameter.numel() for parameter in initial_model.parameters())
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError("initial acceptance model parameter count changed")
    initial_state_hash = state_dict_sha256(initial_model)
    initial_metrics = validate(
        initial_model,
        records,
        device,
        patch_batch_size=args.patch_batch_size,
        cache=cache,
    )
    del initial_model
    torch.cuda.empty_cache()

    final_model = ContextualPairFusionAssociationModel().to(device)
    fold_source = pretraining["folds"][args.fold]
    final_model.load_state_dict(
        torch.load(
            fold_source["model_path"], map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    final_metrics = validate(
        final_model,
        records,
        device,
        patch_batch_size=args.patch_batch_size,
        cache=cache,
    )
    gate = validation_improvement_gate(initial_metrics, final_metrics)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "fold": args.fold,
        "seed": seed,
        "elapsed_seconds": time.monotonic() - started,
        "appearance_family": CONTEXTUAL_PAIR_FUSION_FAMILY,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "acceptance_source": ACCEPTANCE_SOURCE,
        "acceptance_role": ACCEPTANCE_ROLE,
        "acceptance_shards": len(records),
        "acceptance_cache_bytes": cache_bytes,
        "acceptance_manifest_sha256": acceptance_evidence["manifest_sha256"],
        "acceptance_inventory_sha256": acceptance_evidence["inventory_sha256"],
        "acceptance_record_inventory_sha256": inventory_sha256(records),
        "pretraining_terminal_sha256": pretraining["terminal_sha256"],
        "pretrained_model_sha256": fold_source["model_sha256"],
        "pretraining_worker_terminal_sha256": fold_source[
            "worker_terminal_sha256"
        ],
        "initial_state_sha256": initial_state_hash,
        "initial": initial_metrics,
        "final": final_metrics,
        "gate": gate,
        "gate_passed": bool(gate["passed"]),
        "selection_or_checkpoint_redirect_permitted": False,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(terminal_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def orchestrate(args: argparse.Namespace) -> None:
    if not 0 < args.hard_stop_seconds <= DEFAULT_HARD_STOP_SECONDS:
        raise ValueError("acceptance hard stop must be in (0, 3600]")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"acceptance evaluation requires exactly two GPUs, saw {torch.cuda.device_count()}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("one-shot acceptance output directory is not empty")
    pretraining = verify_pretraining_source(args.pretraining_root)
    acceptance_evidence, _records = verify_acceptance_source(args.acceptance_root)
    tokens = visible_cuda_tokens(detected_devices=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    processes = []
    for index, fold in enumerate(FOLDS):
        log_path = args.output_dir / f"{fold}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--fold",
            fold,
            "--pretraining-root",
            str(args.pretraining_root),
            "--acceptance-root",
            str(args.acceptance_root),
            "--output-dir",
            str(args.output_dir),
            "--patch-batch-size",
            str(args.patch_batch_size),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = tokens[index]
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((fold, process, handle, log_path))
    return_codes: dict[str, int] = {}
    try:
        while len(return_codes) < len(processes):
            for fold, process, _handle, _log in processes:
                code = process.poll()
                if code is not None and fold not in return_codes:
                    return_codes[fold] = int(code)
            if time.monotonic() - started >= args.hard_stop_seconds:
                raise TimeoutError("two-GPU ZSNS001 acceptance exceeded hard stop")
            if len(return_codes) < len(processes):
                time.sleep(2)
    finally:
        terminate_and_reap_processes(
            [process for _fold, process, _handle, _log in processes]
        )
        for _fold, _process, handle, _log in processes:
            handle.close()
    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"ZSNS001 acceptance workers failed: {failures}")
    folds: dict[str, Any] = {}
    for fold in FOLDS:
        path = args.output_dir / fold / "acceptance_terminal.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if not (
            row.get("status") == "completed"
            and row.get("fold") == fold
            and row.get("pretrained_model_sha256")
            == pretraining["folds"][fold]["model_sha256"]
            and row.get("acceptance_manifest_sha256")
            == EXPECTED_ACCEPTANCE_MANIFEST_SHA256
            and row.get("acceptance_inventory_sha256")
            == EXPECTED_ACCEPTANCE_INVENTORY_SHA256
            and row.get("acceptance_record_inventory_sha256")
            == EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256
            and row.get("selection_or_checkpoint_redirect_permitted") is False
            and row.get("competition_data_read") is False
            and row.get("public_predictions_copied") is False
            and row.get("public_leaderboard_used_for_selection") is False
            and row.get("submission_created") is False
        ):
            raise ValueError(f"invalid ZSNS001 acceptance terminal: {fold}")
        row["terminal_sha256"] = sha256_file(path)
        folds[fold] = row
    payload = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "pretraining_run_id": PRETRAINING_RUN_ID,
        "pretraining_terminal_sha256": pretraining["terminal_sha256"],
        "acceptance_source": ACCEPTANCE_SOURCE,
        "acceptance_manifest_sha256": acceptance_evidence["manifest_sha256"],
        "acceptance_inventory_sha256": acceptance_evidence["inventory_sha256"],
        "acceptance_record_inventory_sha256": (
            EXPECTED_ACCEPTANCE_RECORD_INVENTORY_SHA256
        ),
        "both_folds_improved": all(row["gate_passed"] is True for row in folds.values()),
        "folds": folds,
        "selection_or_checkpoint_redirect_permitted": False,
        "competition_data_read": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
    }
    atomic_json(args.output_dir / "acceptance_terminal.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--fold", choices=FOLDS)
    result.add_argument("--pretraining-root", type=Path, required=True)
    result.add_argument("--acceptance-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--patch-batch-size", type=int, default=48)
    result.add_argument(
        "--hard-stop-seconds", type=int, default=DEFAULT_HARD_STOP_SECONDS
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.patch_batch_size <= 0:
        raise ValueError("patch batch size must be positive")
    if args.worker:
        if args.fold is None:
            raise ValueError("--fold is required for an acceptance worker")
        evaluate_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
