"""Two-GPU, leave-one-embryo-out Trackastra adaptation for Biohub.

The public synthetic sequence images are XY pooled, while their graph nodes are
stored in native Biohub coordinates.  ``corrected_sequence_graph`` repairs the
nodes for image access.  A geometry-only Trackastra model instead expects the
native coordinate convention used by the real GEFF graphs, so this trainer
explicitly restores Y/X after validating the pooled representation.

Each GPU owns one reciprocal fold.  The late training state is never selected
implicitly: a fixed, unopened opposite-embryo ranking set chooses a checkpoint,
and the official pretrained initialization remains a valid candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    import trainer as base
    from synthetic_data import DEFAULT_XY_STRIDE, corrected_sequence_graph, division_prior_weight
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    import research.trackastra_graph.train_biohub_graph_transformer as base
    from research.synthetic_pretrain.data import (
        DEFAULT_XY_STRIDE,
        corrected_sequence_graph,
        division_prior_weight,
    )


RUN_ID = "trackastra-dual-fold-synthetic-v1"
OPENED_ACCEPTANCE_STEMS = frozenset(
    {
        "44b6_12dfb391",
        "44b6_267148e4",
        "6bba_062c8d37",
        "6bba_07e24132",
    }
)
FOLD_SPECS = {
    "target_44b6": {"target_prefix": "44b6", "source_prefix": "6bba", "seed_offset": 4400},
    "target_6bba": {"target_prefix": "6bba", "source_prefix": "44b6", "seed_offset": 6600},
}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_video_native_geometry(
    path: Path, *, xy_stride: int = DEFAULT_XY_STRIDE
) -> base.GraphVideo:
    """Load a validated synthetic graph in Trackastra's native voxel frame."""

    graph = corrected_sequence_graph(path, xy_stride=xy_stride)
    native_coords = np.asarray(graph.nodes[:, 1:4], dtype=np.float32).copy()
    native_coords[:, 1:] *= float(xy_stride)
    node_ids = np.arange(len(graph.nodes), dtype=np.int64)
    return base.GraphVideo(
        stem=f"synthetic_native_{path.stem}",
        node_ids=node_ids,
        times=graph.nodes[:, 0].astype(np.int32),
        coords_voxel=native_coords,
        edges=graph.edges,
    )


def split_synthetic_paths(
    root: Path, *, validation_count: int, train_limit: int | None = None
) -> tuple[Path, list[Path], list[Path]]:
    manifest, ranked = base.select_synthetic_sequence_paths(root, limit=10_000_000)
    if validation_count <= 0 or validation_count >= len(ranked):
        raise ValueError("synthetic validation count must leave non-empty train and validation sets")
    validation = ranked[:validation_count]
    training = ranked[validation_count:]
    if train_limit is not None:
        if train_limit <= 0:
            raise ValueError("synthetic train limit must be positive")
        training = training[:train_limit]
    return manifest, training, validation


def select_real_paths(
    train_dir: Path,
    *,
    prefix: str,
    excluded_stems: Iterable[str] = OPENED_ACCEPTANCE_STEMS,
    limit: int | None = None,
) -> list[Path]:
    excluded = set(excluded_stems)
    paths = [
        path
        for path in train_dir.glob(f"{prefix}_*.geff")
        if path.stem not in excluded
    ]
    ranked = sorted(
        paths,
        key=lambda path: hashlib.sha256(f"dual-fold-v1:{path.stem}".encode("utf-8")).hexdigest(),
    )
    if limit is not None:
        ranked = ranked[:limit]
    if not ranked:
        raise FileNotFoundError(f"no eligible real graphs for prefix {prefix}")
    if any(path.stem in OPENED_ACCEPTANCE_STEMS for path in ranked):
        raise RuntimeError("opened acceptance labels leaked into reciprocal-fold data")
    return ranked


def augment_global_motion(
    sample: base.WindowSample,
    rng: np.random.Generator,
    *,
    probability: float,
    step_sigma: float,
    jump_probability: float,
    jump_sigma: float,
) -> base.WindowSample:
    """Add frame-global drift/jumps while keeping graph targets unchanged."""

    if rng.random() >= probability:
        return sample
    coords = sample.coords.copy()
    times = coords[:, 0].astype(np.int32)
    unique_times = np.unique(times)
    offsets: dict[int, np.ndarray] = {}
    current = np.zeros(3, dtype=np.float32)
    for timepoint in sorted(unique_times.tolist()):
        current = current + rng.normal(0.0, step_sigma, 3).astype(np.float32)
        if rng.random() < jump_probability:
            current = current + rng.normal(0.0, jump_sigma, 3).astype(np.float32)
        offsets[int(timepoint)] = current.copy()
    for timepoint, offset in offsets.items():
        coords[times == timepoint, 1:] += offset[None]
    features = base.point_features(coords, times)
    return replace(sample, coords=coords, features=features)


def fixed_validation_samples(
    videos: list[base.GraphVideo],
    *,
    seed: int,
    count: int,
    sample_kwargs: dict[str, Any],
) -> list[base.WindowSample]:
    rng = np.random.default_rng(seed)
    clean_kwargs = {
        **sample_kwargs,
        "drop_probability": 0.0,
        "false_positive_probability": 0.0,
        "false_positive_ratio": 0.0,
        "jitter_sigma": 0.0,
        "prefer_division_probability": 1.0,
    }
    return [base.sample_window(videos, rng, **clean_kwargs) for _ in range(count)]


@torch.no_grad()
def ranking_metrics(
    model: torch.nn.Module,
    samples: list[base.WindowSample],
    device: torch.device,
) -> dict[str, float | int]:
    """Threshold-free parent and division ranking on fixed graph windows."""

    model.eval()
    reciprocal_ranks: list[float] = []
    top1: list[float] = []
    margins: list[float] = []
    division_recalls: list[float] = []
    division_exact: list[float] = []
    for sample in samples:
        coords = torch.from_numpy(sample.coords).unsqueeze(0).to(device)
        features = torch.from_numpy(sample.features).unsqueeze(0).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(coords, features)
        timepoints = coords[:, :, 0].long()
        probabilities = (
            model.normalize_output(logits.float(), timepoints, coords.float())[0]
            .detach()
            .cpu()
            .numpy()
        )
        for source in np.flatnonzero(sample.target.sum(axis=1) > 0):
            candidates = np.flatnonzero(sample.valid_mask[source])
            positives = set(np.flatnonzero(sample.target[source] > 0).tolist())
            if not len(candidates) or not positives:
                continue
            ordered = candidates[
                np.argsort(-probabilities[source, candidates], kind="stable")
            ]
            positive_ranks = [
                rank for rank, target in enumerate(ordered.tolist(), start=1) if target in positives
            ]
            if not positive_ranks:
                continue
            reciprocal_ranks.append(1.0 / min(positive_ranks))
            top1.append(float(int(ordered[0]) in positives))
            positive_score = max(probabilities[source, target] for target in positives)
            negative_scores = [
                probabilities[source, target]
                for target in candidates.tolist()
                if target not in positives
            ]
            negative_score = max(negative_scores) if negative_scores else 0.0
            margins.append(float(positive_score - negative_score))

            division_targets = set(
                np.flatnonzero(sample.division_target[source] > 0).tolist()
            )
            if len(division_targets) >= 2:
                predicted_two = set(map(int, ordered[:2].tolist()))
                recovered = len(predicted_two & division_targets) / len(division_targets)
                division_recalls.append(float(recovered))
                division_exact.append(float(division_targets.issubset(predicted_two)))

    if not reciprocal_ranks:
        raise RuntimeError("fixed validation produced no rankable positive edges")
    edge_top1 = float(np.mean(top1))
    edge_mrr = float(np.mean(reciprocal_ranks))
    mean_margin = float(np.mean(margins))
    margin_score = float(np.clip((mean_margin + 1.0) / 2.0, 0.0, 1.0))
    division_top2 = float(np.mean(division_recalls)) if division_recalls else 0.0
    division_both = float(np.mean(division_exact)) if division_exact else 0.0
    edge_component = 0.55 * edge_top1 + 0.35 * edge_mrr + 0.10 * margin_score
    if division_recalls:
        composite = 0.80 * edge_component + 0.12 * division_top2 + 0.08 * division_both
    else:
        composite = edge_component
    model.train()
    return {
        "composite": composite,
        "edge_top1_accuracy": edge_top1,
        "edge_mrr": edge_mrr,
        "mean_positive_margin": mean_margin,
        "division_top2_recall": division_top2,
        "division_both_top2": division_both,
        "edge_rows": len(reciprocal_ranks),
        "division_rows": len(division_recalls),
        "samples": len(samples),
    }


def selection_score(real_metrics: dict[str, Any], synthetic_metrics: dict[str, Any]) -> float:
    return 0.85 * float(real_metrics["composite"]) + 0.15 * float(
        synthetic_metrics["composite"]
    )


def state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def validation_schedule(step: int, target_steps: int) -> bool:
    return step in {500, 1500, 3000} or step % 5000 == 0 or step == target_steps


def train_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if args.fold not in FOLD_SPECS:
        raise ValueError(f"unknown fold {args.fold}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"isolated worker must see exactly one CUDA device, saw {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    spec = FOLD_SPECS[args.fold]
    seed = args.seed + int(spec["seed_offset"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trackastra_dir = args.trackastra_dir.resolve()
    sys.path.insert(0, str(trackastra_dir))
    from trackastra.model.model import TrackingTransformer

    output_dir = args.output_dir / args.fold
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.competition_dir / "train"

    manifest, synthetic_train_paths, synthetic_validation_paths = split_synthetic_paths(
        args.synthetic_root,
        validation_count=args.synthetic_validation_graphs,
        train_limit=args.synthetic_train_graphs,
    )
    real_train_paths = select_real_paths(
        train_dir, prefix=str(spec["source_prefix"]), limit=args.real_train_graphs
    )
    real_validation_paths = select_real_paths(
        train_dir, prefix=str(spec["target_prefix"]), limit=args.real_validation_graphs
    )
    print(
        f"{args.fold}: loading {len(synthetic_train_paths)} synthetic train, "
        f"{len(synthetic_validation_paths)} synthetic validation, "
        f"{len(real_train_paths)} real source, {len(real_validation_paths)} real target graphs",
        flush=True,
    )
    synthetic_train = [synthetic_video_native_geometry(path) for path in synthetic_train_paths]
    synthetic_validation = [
        synthetic_video_native_geometry(path) for path in synthetic_validation_paths
    ]
    real_train = [base.read_graph_video(path) for path in real_train_paths]
    real_validation = [base.read_graph_video(path) for path in real_validation_paths]

    model = TrackingTransformer.from_folder(args.pretrained_dir, map_location="cpu").to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count < 20_000_000:
        raise RuntimeError(f"unexpected Trackastra parameter count {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    sample_kwargs = {
        "window": 4,
        "max_tokens": args.max_tokens,
        "tile_radius": np.array((96.0, 192.0, 192.0), dtype=np.float32),
        "drop_probability": args.drop_probability,
        "false_positive_probability": 0.0,
        "false_positive_ratio": args.false_positive_ratio,
        "false_positive_uniform_fraction": 0.4,
        "false_positive_local_sigma": 12.0,
        "false_positive_min_distance": 4.0,
        "jitter_sigma": args.jitter_sigma,
        "prefer_division_probability": args.prefer_division_probability,
        "hard_negative_radius": args.hard_negative_radius,
    }
    real_fixed = fixed_validation_samples(
        real_validation,
        seed=seed + 101,
        count=args.real_validation_samples,
        sample_kwargs=sample_kwargs,
    )
    synthetic_fixed = fixed_validation_samples(
        synthetic_validation,
        seed=seed + 202,
        count=args.synthetic_validation_samples,
        sample_kwargs=sample_kwargs,
    )

    initial_real = ranking_metrics(model, real_fixed, device)
    initial_synthetic = ranking_metrics(model, synthetic_fixed, device)
    initial_score = selection_score(initial_real, initial_synthetic)
    best_score = initial_score
    best_step = 0
    best_real = initial_real
    best_synthetic = initial_synthetic
    best_state = state_dict_cpu(model)
    validation_rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "selection_score": initial_score,
            "eligible": True,
            "real": initial_real,
            "synthetic": initial_synthetic,
        }
    ]
    atomic_json(output_dir / "initial_validation.json", validation_rows[0])

    config = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "fold": args.fold,
        "seed": seed,
        "target_prefix": spec["target_prefix"],
        "source_prefix": spec["source_prefix"],
        "opened_acceptance_stems_excluded": sorted(OPENED_ACCEPTANCE_STEMS),
        "parameter_count": parameter_count,
        "pretrained_sha256": sha256_file(args.pretrained_dir / "model.pt"),
        "synthetic_manifest_sha256": sha256_file(manifest),
        "synthetic_train_count": len(synthetic_train_paths),
        "synthetic_validation_count": len(synthetic_validation_paths),
        "synthetic_train_names_sha256": hashlib.sha256(
            "\n".join(path.name for path in synthetic_train_paths).encode("utf-8")
        ).hexdigest(),
        "synthetic_validation_names_sha256": hashlib.sha256(
            "\n".join(path.name for path in synthetic_validation_paths).encode("utf-8")
        ).hexdigest(),
        "real_train_stems": [path.stem for path in real_train_paths],
        "real_validation_stems": [path.stem for path in real_validation_paths],
        "synthetic_native_geometry_restored": True,
        "steps_target": args.steps,
        "learning_rate": args.learning_rate,
        "real_replay_probability": args.real_replay_probability,
        "division_prior_weight": division_prior_weight(),
        "selection": "0.85 opposite-embryo ranking + 0.15 heldout-synthetic ranking",
    }
    atomic_json(output_dir / "training_config.json", config)

    rng = np.random.default_rng(seed + 303)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rolling_losses: list[float] = []
    history_path = output_dir / "history.jsonl"
    stopped_for_time = False
    completed_step = 0
    for step in range(1, args.steps + 1):
        elapsed = time.monotonic() - started
        if elapsed >= args.max_wall_seconds - args.finalization_reserve_seconds:
            stopped_for_time = True
            print(f"{args.fold}: stopping at step {step - 1} for finalization reserve", flush=True)
            break
        use_real = bool(rng.random() < args.real_replay_probability)
        videos = real_train if use_real else synthetic_train
        sample = base.sample_window(videos, rng, **sample_kwargs)
        sample = augment_global_motion(
            sample,
            rng,
            probability=args.global_motion_probability,
            step_sigma=args.global_motion_step_sigma,
            jump_probability=args.global_jump_probability,
            jump_sigma=args.global_jump_sigma,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, stats = base.association_loss(
                model,
                sample,
                device,
                division_weight_scale=(1.0 if use_real else division_prior_weight()),
            )
            scaled_loss = loss / args.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        if step % args.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        completed_step = step
        rolling_losses.append(float(loss.item()))

        progress = min(step / max(args.steps, 1), 1.0)
        learning_rate = args.min_learning_rate + 0.5 * (
            args.learning_rate - args.min_learning_rate
        ) * (1.0 + np.cos(np.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)

        if step % args.log_every == 0:
            row = {
                "step": step,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "learning_rate": float(learning_rate),
                "mean_loss": float(np.mean(rolling_losses[-args.log_every :])),
                "source": "real" if use_real else "synthetic",
                **stats,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_plain(row), sort_keys=True) + "\n")
            print(f"{args.fold}: TRAIN {row}", flush=True)

        if validation_schedule(step, args.steps):
            real_metrics = ranking_metrics(model, real_fixed, device)
            synthetic_metrics = ranking_metrics(model, synthetic_fixed, device)
            score = selection_score(real_metrics, synthetic_metrics)
            eligible = bool(
                float(real_metrics["composite"])
                >= float(initial_real["composite"]) + args.minimum_real_gain
                and float(synthetic_metrics["composite"])
                >= float(initial_synthetic["composite"]) - args.maximum_synthetic_regression
            )
            validation_row = {
                "step": step,
                "selection_score": score,
                "eligible": eligible,
                "real": real_metrics,
                "synthetic": synthetic_metrics,
            }
            validation_rows.append(validation_row)
            atomic_json(output_dir / "validation_latest.json", validation_row)
            if eligible and score > best_score:
                best_score = score
                best_step = step
                best_real = real_metrics
                best_synthetic = synthetic_metrics
                best_state = state_dict_cpu(model)
                print(f"{args.fold}: NEW BEST {validation_row}", flush=True)
            model.train()

    if completed_step % args.gradient_accumulation:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    model.load_state_dict(best_state)
    model.eval()
    model_path = output_dir / "model.pt"
    temporary_model = model_path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary_model)
    temporary_model.replace(model_path)
    shutil.copy2(args.pretrained_dir / "config.yaml", output_dir / "config.yaml")
    atomic_json(output_dir / "validation_history.json", {"rows": validation_rows})
    terminal = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "fold": args.fold,
        "elapsed_seconds": time.monotonic() - started,
        "stopped_for_time": stopped_for_time,
        "completed_step": completed_step,
        "best_step": best_step,
        "pretrained_initialization_retained": best_step == 0,
        "initial_selection_score": initial_score,
        "best_selection_score": best_score,
        "initial_real": initial_real,
        "best_real": best_real,
        "initial_synthetic": initial_synthetic,
        "best_synthetic": best_synthetic,
        "model_sha256": sha256_file(model_path),
        "parameter_count": parameter_count,
        "submission_created": False,
    }
    atomic_json(output_dir / "worker_terminal.json", terminal)
    print(json.dumps(_plain(terminal), indent=2), flush=True)


def orchestrate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"two-GPU contract requires exactly 2 visible CUDA devices, saw {torch.cuda.device_count()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for gpu_index, fold in enumerate(FOLD_SPECS):
        log_path = args.output_dir / f"{fold}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
            "--fold",
            fold,
        ]
        command = [item for item in command if item != "--orchestrate"]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((fold, process, log_handle))

    return_codes: dict[str, int] = {}
    try:
        while len(return_codes) < len(processes):
            for fold, process, _handle in processes:
                code = process.poll()
                if code is not None and fold not in return_codes:
                    return_codes[fold] = int(code)
                    print(f"{fold}: worker exited {code}", flush=True)
            if time.monotonic() - started >= args.orchestrator_hard_stop_seconds:
                raise TimeoutError("dual-fold orchestrator exceeded its hard stop")
            if len(return_codes) < len(processes):
                time.sleep(5)
    finally:
        for _fold, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()

    failures = {fold: code for fold, code in return_codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"dual-fold workers failed: {failures}")
    terminals = {
        fold: json.loads((args.output_dir / fold / "worker_terminal.json").read_text(encoding="utf-8"))
        for fold in FOLD_SPECS
    }
    aggregate = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "elapsed_seconds": time.monotonic() - started,
        "gpu_count": 2,
        "whole_fold_coverage": sorted(terminals),
        "folds": terminals,
        "both_folds_improved": all(
            int(terminal["best_step"]) > 0 for terminal in terminals.values()
        ),
        "submission_created": False,
    }
    atomic_json(args.output_dir / "training_terminal.json", aggregate)
    print(json.dumps(_plain(aggregate), indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    role = result.add_mutually_exclusive_group(required=True)
    role.add_argument("--orchestrate", action="store_true")
    role.add_argument("--worker", action="store_true")
    result.add_argument("--fold", choices=sorted(FOLD_SPECS))
    result.add_argument("--competition-dir", type=Path, required=True)
    result.add_argument("--trackastra-dir", type=Path, required=True)
    result.add_argument("--pretrained-dir", type=Path, required=True)
    result.add_argument("--synthetic-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=29011)
    result.add_argument("--steps", type=int, default=75000)
    result.add_argument("--max-wall-seconds", type=int, default=12000)
    result.add_argument("--orchestrator-hard-stop-seconds", type=int, default=13200)
    result.add_argument("--finalization-reserve-seconds", type=int, default=900)
    result.add_argument("--synthetic-train-graphs", type=int, default=2046)
    result.add_argument("--synthetic-validation-graphs", type=int, default=128)
    result.add_argument("--real-train-graphs", type=int, default=96)
    result.add_argument("--real-validation-graphs", type=int, default=12)
    result.add_argument("--real-validation-samples", type=int, default=64)
    result.add_argument("--synthetic-validation-samples", type=int, default=32)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--learning-rate", type=float, default=8e-6)
    result.add_argument("--min-learning-rate", type=float, default=5e-7)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--gradient-accumulation", type=int, default=4)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--drop-probability", type=float, default=0.04)
    result.add_argument("--false-positive-ratio", type=float, default=0.75)
    result.add_argument("--jitter-sigma", type=float, default=1.0)
    result.add_argument("--prefer-division-probability", type=float, default=0.15)
    result.add_argument("--hard-negative-radius", type=float, default=48.0)
    result.add_argument("--real-replay-probability", type=float, default=0.05)
    result.add_argument("--global-motion-probability", type=float, default=0.35)
    result.add_argument("--global-motion-step-sigma", type=float, default=1.5)
    result.add_argument("--global-jump-probability", type=float, default=0.08)
    result.add_argument("--global-jump-sigma", type=float, default=8.0)
    result.add_argument("--minimum-real-gain", type=float, default=0.001)
    result.add_argument("--maximum-synthetic-regression", type=float, default=0.005)
    result.add_argument("--log-every", type=int, default=100)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.fold is None:
            raise ValueError("--fold is required for a worker")
        train_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
