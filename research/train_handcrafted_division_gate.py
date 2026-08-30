#!/usr/bin/env python
"""Train a deterministic CPU morphology control on real Biohub division patches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_ID = "competition-real-handcrafted-division-gate-v1"
FEATURE_FAMILY = "temporal_radial_peak_morphology_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _peak_features(volume: np.ndarray) -> list[float]:
    side = volume.shape[0]
    coordinates = np.stack(
        np.meshgrid(*(np.arange(side, dtype=np.float64) for _ in range(3)), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    center = np.asarray([(side - 1) / 2.0] * 3)
    flattened = volume.reshape(-1)
    first_index = int(np.argmax(flattened))
    first = coordinates[first_index]
    distances = np.linalg.norm(coordinates - first, axis=1)
    eligible = np.flatnonzero(distances >= 3.0)
    second_index = int(eligible[np.argmax(flattened[eligible])])
    second = coordinates[second_index]
    first_value = float(flattened[first_index])
    second_value = float(flattened[second_index])
    first_step = first - center
    second_step = second - center
    denominator = max(float(np.linalg.norm(first_step) * np.linalg.norm(second_step)), 1e-8)
    return [
        first_value,
        second_value,
        second_value / max(abs(first_value), 1e-6),
        float(np.linalg.norm(first - second)),
        float(np.linalg.norm(0.5 * (first + second) - center)),
        float(np.dot(first_step, second_step) / denominator),
    ]


def patch_features(patches: np.ndarray) -> np.ndarray:
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 5 or patches.shape[1:] != (3, 17, 17, 17):
        raise ValueError("handcrafted division patches must have shape (N,3,17,17,17)")
    axis = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.sqrt(zz * zz + yy * yy + xx * xx)
    shells = (
        radius <= 0.25,
        (radius > 0.25) & (radius <= 0.50),
        (radius > 0.50) & (radius <= 0.75),
        (radius > 0.75) & (radius <= 1.00),
        (radius > 1.00) & (radius <= 1.35),
    )
    rows: list[list[float]] = []
    for patch in patches:
        features: list[float] = []
        for channel in patch:
            quantiles = np.quantile(channel, (0.50, 0.75, 0.90, 0.95, 0.99))
            gradients = np.gradient(channel.astype(np.float64, copy=False))
            features.extend(float(value) for value in quantiles)
            features.extend(
                [
                    float(channel.min()),
                    float(channel.max()),
                    float(np.mean(np.abs(channel))),
                    *(float(np.mean(value * value)) for value in gradients),
                ]
            )
            for shell in shells:
                values = channel[shell]
                features.extend(
                    (float(values.mean()), float(values.std()), float(values.max()))
                )
            bright = np.maximum(channel - quantiles[2], 0.0).astype(np.float64)
            total = float(bright.sum())
            if total <= 1e-8:
                centroid = np.zeros(3)
                eigenvalues = np.zeros(3)
            else:
                coordinates = np.stack((zz, yy, xx), axis=-1)
                centroid = (coordinates * bright[..., None]).sum(axis=(0, 1, 2)) / total
                centered = coordinates - centroid
                covariance = np.einsum(
                    "...i,...j,...->ij", centered, centered, bright, optimize=True
                ) / total
                eigenvalues = np.linalg.eigvalsh(covariance)
            features.extend(float(value) for value in centroid)
            features.extend(float(value) for value in eigenvalues)
            features.extend(_peak_features(channel))
        flattened = [patch[index].reshape(-1) for index in range(3)]
        for left, right in ((0, 1), (1, 2), (0, 2)):
            left_std = float(flattened[left].std())
            right_std = float(flattened[right].std())
            correlation = (
                float(np.corrcoef(flattened[left], flattened[right])[0, 1])
                if min(left_std, right_std) > 1e-8
                else 0.0
            )
            features.append(correlation)
            difference = flattened[right] - flattened[left]
            features.extend(
                [
                    float(np.mean(np.abs(difference))),
                    float(np.sqrt(np.mean(difference * difference))),
                    *(float(value) for value in np.quantile(difference, (0.05, 0.5, 0.95))),
                ]
            )
        rows.append(features)
    result = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(result).all() or result.shape[1] != 132:
        raise RuntimeError(f"handcrafted division feature contract changed: {result.shape}")
    return result


def load_role(
    root: Path, manifest: dict[str, Any], role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patches = []
    targets = []
    weights = []
    groups = []
    for record in manifest["shards"]:
        if record["role"] != role:
            continue
        path = root / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"handcrafted division shard changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            source = np.asarray(data["source_patches"], dtype=np.float32)
            target = np.asarray(data["division_target"], dtype=np.int64)
            weight = np.asarray(data["label_weight"], dtype=np.float64)
        patches.append(source)
        targets.append(target)
        weights.append(weight)
        groups.extend([record["stem"]] * len(target))
    return (
        patch_features(np.concatenate(patches)),
        np.concatenate(targets),
        np.concatenate(weights),
        np.asarray(groups),
    )


def high_precision_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, kind="stable")
    labels = targets[order].astype(bool)
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    positives = int(targets.sum())
    recall = tp / positives
    zero_fp = np.flatnonzero((fp == 0) & (tp > 0))
    precise = np.flatnonzero((tp / np.maximum(tp + fp, 1) >= 0.95) & (tp > 0))
    return {
        "rows": len(targets),
        "positives": positives,
        "average_precision": float(average_precision_score(targets, scores)),
        "recall_at_zero_false_positives": (
            float(recall[zero_fp[-1]]) if len(zero_fp) else 0.0
        ),
        "recall_at_precision_0_95": (
            float(recall[precise].max()) if len(precise) else 0.0
        ),
    }


def estimator_grid(seed: int) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for value in (0.01, 0.1, 1.0, 10.0):
        rows.append(
            (
                f"logistic_c{value:g}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=value, max_iter=3000, random_state=seed),
                ),
            )
        )
    for leaf in (2, 5, 10):
        rows.append(
            (
                f"extra_trees_leaf{leaf}",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=leaf,
                    max_features=0.75,
                    n_jobs=-1,
                    random_state=seed,
                ),
            )
        )
    for leaf in (10, 20):
        rows.append(
            (
                f"hist_gradient_leaf{leaf}",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=leaf,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            )
        )
    return rows


def fit_with_weights(estimator: Any, features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> Any:
    if hasattr(estimator, "steps"):
        final_name = estimator.steps[-1][0]
        return estimator.fit(features, targets, **{f"{final_name}__sample_weight": weights})
    return estimator.fit(features, targets, sample_weight=weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=126_071)
    args = parser.parse_args()
    started = time.monotonic()
    manifest_path = args.data_root / "real_division_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("run_id") == "competition-real-division-patches-v1"
        and manifest.get("competition_test_data_read") is False
        and manifest.get("summary", {}).get("division_positives") == 146
    ):
        raise ValueError("handcrafted division manifest is ineligible")
    args.output_root.mkdir(parents=True, exist_ok=False)
    train_x, train_y, train_weights, groups = load_role(
        args.data_root, manifest, "optimization"
    )
    selection_x, selection_y, _, _ = load_role(args.data_root, manifest, "selection")
    class_balance = float((train_weights * (train_y == 0)).sum() / (train_y == 1).sum())
    fit_weights = train_weights * np.where(train_y == 1, class_balance, 1.0)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = []
    for name, estimator in estimator_grid(args.seed):
        oof = np.zeros(len(train_y), dtype=np.float64)
        for fit, score in splitter.split(train_x, train_y, groups):
            model = fit_with_weights(
                clone(estimator), train_x[fit], train_y[fit], fit_weights[fit]
            )
            oof[score] = model.predict_proba(train_x[score])[:, 1]
        candidates.append(
            {
                "name": name,
                "estimator": estimator,
                "oof_scores": oof,
                "oof": high_precision_metrics(train_y, oof),
            }
        )
        print(json.dumps({"name": name, "oof": candidates[-1]["oof"]}), flush=True)
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["oof"]["recall_at_zero_false_positives"],
            row["oof"]["recall_at_precision_0_95"],
            row["oof"]["average_precision"],
        ),
        reverse=True,
    )
    selected = ranked[:2]
    fitted = []
    selection_parts = []
    for row in selected:
        model = fit_with_weights(
            clone(row["estimator"]), train_x, train_y, fit_weights
        )
        fitted.append({"name": row["name"], "model": model})
        selection_parts.append(model.predict_proba(selection_x)[:, 1])
    ensemble_scores = np.mean(np.stack(selection_parts), axis=0)
    selection_metrics = high_precision_metrics(selection_y, ensemble_scores)
    model_path = args.output_root / "handcrafted_division_gate.joblib"
    joblib.dump(
        {
            "run_id": RUN_ID,
            "feature_family": FEATURE_FAMILY,
            "feature_count": train_x.shape[1],
            "models": fitted,
        },
        model_path,
        compress=3,
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "run_id": RUN_ID,
        "feature_family": FEATURE_FAMILY,
        "feature_count": train_x.shape[1],
        "elapsed_seconds": time.monotonic() - started,
        "train_rows": len(train_y),
        "train_positives": int(train_y.sum()),
        "selection_rows": len(selection_y),
        "selection_positives": int(selection_y.sum()),
        "cross_validation": [
            {"name": row["name"], "metrics": row["oof"]} for row in ranked
        ],
        "selected_models": [row["name"] for row in selected],
        "selection": selection_metrics,
        "model_sha256": sha256_file(model_path),
        "manifest_sha256": sha256_file(manifest_path),
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "final_probe_opened": False,
        "public_code_copied": False,
        "public_predictions_copied": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output_root / "handcrafted_division_gate_terminal.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
