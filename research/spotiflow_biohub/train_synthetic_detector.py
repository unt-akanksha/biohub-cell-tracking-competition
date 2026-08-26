#!/usr/bin/env python
"""Fine-tune an official Spotiflow 3D detector on corrected Biohub synthetic data.

The public synthetic kernel stores native 64x256x256 static volumes and native
centroids.  This trainer pools image and point coordinates together, streams the
NPZ files, and keeps Spotiflow's repeated input-validation passes metadata-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from synthetic_data import (
        PointSequence,
        PooledImageSequence,
        SyntheticStaticStore,
        split_static_paths,
    )
except ModuleNotFoundError:
    from research.synthetic_pretrain.data import (
        PointSequence,
        PooledImageSequence,
        SyntheticStaticStore,
        split_static_paths,
    )


EXPECTED_NATIVE_SHAPE = (64, 256, 256)
EXPECTED_POOLED_SHAPE = (64, 64, 64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_manifest(root: Path) -> Path:
    direct = root / "manifest.json"
    candidates = [direct] if direct.is_file() else sorted(root.rglob("manifest.json"))
    valid = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("static"):
            valid.append(path)
    if len(valid) != 1:
        raise FileNotFoundError(
            f"Expected one synthetic manifest below {root}, found {[str(p) for p in valid]}"
        )
    return valid[0]


def load_static_records(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("static")
    if not isinstance(records, list) or not records:
        raise ValueError("synthetic manifest has no static records")
    normalized = []
    for record in records:
        relative = Path(record["file"])
        shape = tuple(int(value) for value in record["shape"])
        count = int(record["n_cells"])
        path = manifest_path.parent / relative
        if shape != EXPECTED_NATIVE_SHAPE:
            raise ValueError(f"unexpected synthetic shape for {relative}: {shape}")
        if count <= 0 or not path.is_file():
            raise ValueError(f"invalid synthetic record: {record}")
        normalized.append({"path": path, "n_cells": count, "shape": shape})
    return normalized


def split_records(
    records: Sequence[dict[str, Any]], *, validation_count: int, train_limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_path = {Path(record["path"]): record for record in records}
    train_paths, validation_paths = split_static_paths(
        list(by_path), validation_count=validation_count
    )
    if train_limit > 0:
        train_paths = train_paths[:train_limit]
    train = [by_path[path] for path in train_paths]
    validation = [by_path[path] for path in validation_paths]
    if not train or set(train_paths) & set(validation_paths):
        raise RuntimeError("invalid synthetic split")
    return train, validation


def build_sequences(records: Sequence[dict[str, Any]]):
    store = SyntheticStaticStore(
        [record["path"] for record in records],
        pooled_shape=EXPECTED_POOLED_SHAPE,
        point_counts=[int(record["n_cells"]) for record in records],
    )
    return PooledImageSequence(store, lazy=True), PointSequence(store, lazy=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-wall-seconds", type=int, default=6300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if args.epochs <= 0 or args.samples_per_epoch <= 0 or args.max_wall_seconds <= 0:
        raise ValueError("epochs, samples-per-epoch, and max-wall-seconds must be positive")
    if not (0.0 < args.learning_rate <= 1e-3):
        raise ValueError("learning rate is outside the guarded fine-tuning range")

    manifest_path = find_manifest(args.synthetic_root)
    records = load_static_records(manifest_path)
    train_records, validation_records = split_records(
        records,
        validation_count=args.validation_count,
        train_limit=args.train_limit,
    )
    train_images, train_points = build_sequences(train_records)
    validation_images, validation_points = build_sequences(validation_records)

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.loggers import CSVLogger
    from spotiflow.model import Spotiflow
    from spotiflow.utils import normalize

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("this fine-tune lane requires a CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "training_progress.json"

    class ProgressAndBudget(pl.Callback):
        def _metrics(self, trainer) -> dict[str, float]:
            metrics = {}
            for key, value in trainer.callback_metrics.items():
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metrics[str(key)] = float(value)
            return metrics

        def _record(self, trainer, stage: str) -> None:
            elapsed = time.monotonic() - started
            write_json(
                progress_path,
                {
                    "stage": stage,
                    "elapsed_seconds": round(elapsed, 3),
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                    "metrics": self._metrics(trainer),
                    "budget_stop_requested": elapsed >= args.max_wall_seconds,
                },
            )
            if elapsed >= args.max_wall_seconds:
                trainer.should_stop = True

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            self._record(trainer, "train_epoch_end")

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.sanity_checking:
                self._record(trainer, "validation_epoch_end")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            if time.monotonic() - started >= args.max_wall_seconds:
                self._record(trainer, "budget_stop_train")

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ) -> None:
            if time.monotonic() - started >= args.max_wall_seconds:
                self._record(trainer, "budget_stop_validation")

    def materialize_normalize(image) -> np.ndarray:
        return normalize(np.asarray(image))

    model = Spotiflow.from_folder(
        str(args.pretrained_model), inference_mode=False, map_location="cpu"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    base_weight = args.pretrained_model / "best.pt"
    if not base_weight.is_file():
        raise FileNotFoundError(base_weight)

    training_config = {
        "crop_size": 64,
        "crop_size_depth": 32,
        "smart_crop": 0.8,
        "num_train_samples": args.samples_per_epoch,
        "lr": float(args.learning_rate),
        "batch_size": 1,
        "lr_reduce_patience": 2,
        "num_epochs": args.epochs,
        "early_stopping_patience": 0,
        "optimize_threshold": False,
        "finetuned_from": str(args.pretrained_model.name),
    }
    run_manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "synthetic_manifest": str(manifest_path),
        "synthetic_manifest_sha256": sha256_file(manifest_path),
        "geometry": {
            "native_shape": EXPECTED_NATIVE_SHAPE,
            "pooled_shape": EXPECTED_POOLED_SHAPE,
            "xy_stride": 4,
            "pooled_voxel_um": [1.625, 1.625, 1.625],
        },
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "train_names_sha256": hashlib.sha256(
            "\n".join(record["path"].name for record in train_records).encode("utf-8")
        ).hexdigest(),
        "validation_names_sha256": hashlib.sha256(
            "\n".join(record["path"].name for record in validation_records).encode("utf-8")
        ).hexdigest(),
        "pretrained_model": str(args.pretrained_model),
        "pretrained_best_sha256": sha256_file(base_weight),
        "parameter_count": parameter_count,
        "training_config": training_config,
        "max_wall_seconds": args.max_wall_seconds,
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))

    logger = CSVLogger(save_dir=str(args.output_dir), name="metrics")
    model.fit(
        train_images,
        train_points,
        validation_images,
        validation_points,
        augment_train=True,
        save_dir=args.output_dir,
        train_config=training_config,
        device="cuda",
        logger=logger,
        number_of_devices=1,
        num_workers=0,
        callbacks=[ProgressAndBudget()],
        deterministic=False,
        benchmark=True,
        default_root_dir=str(args.output_dir),
        defer_normalization=True,
        normalizer=materialize_normalize,
    )

    best_weight = args.output_dir / "best.pt"
    if not best_weight.is_file():
        raise RuntimeError("Spotiflow training ended without a best checkpoint")
    elapsed = time.monotonic() - started
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {}
    )
    result = {
        "status": "completed",
        "elapsed_seconds": round(elapsed, 3),
        "budget_stop_requested": elapsed >= args.max_wall_seconds,
        "best_weight_sha256": sha256_file(best_weight),
        "best_weight_bytes": best_weight.stat().st_size,
        "base_weight_sha256": run_manifest["pretrained_best_sha256"],
        "weights_changed": sha256_file(best_weight) != run_manifest["pretrained_best_sha256"],
        "parameter_count": parameter_count,
        "last_progress": progress,
        "run_manifest_sha256": sha256_file(args.output_dir / "run_manifest.json"),
    }
    write_json(args.output_dir / "synthetic_finetune_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
