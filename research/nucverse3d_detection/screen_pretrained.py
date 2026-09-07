"""Screen the public NucVerse3D generalized checkpoint on train-only crops.

This is an eligibility screen, not a candidate promotion.  It reproduces the
published binary-mask and gradient-attractor decoding constants while using a
hash-selected subset of the optimization partition.  The selection partition
cannot be opened until a matching optimization receipt passes this screen;
the sealed-audit partition is forbidden here.

NucVerse3D source: https://github.com/Segovia-lab/NucVerse3D
Pinned source commit: d809a2e6cf380342708b7a9107574b259e6b34eb
Weights: Zenodo record 18517324, CC-BY-4.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


RUN_ID = "nucverse3d-generalized-physical-compatibility-v1"
SOURCE_COMMIT = "d809a2e6cf380342708b7a9107574b259e6b34eb"
CHECKPOINT_SHA256 = "1c4e288350b1a86d361359cdd02151744d418e9fcf7ebe588c1c77e2cd8bbd67"
ONNX_SHA256 = "ca16e1b26d21ae522d68aba384ee7121f5ba2de28a122c291d1e1e627601e871"
MANIFEST_SHA256 = "5ccd52b96a36db7dff5f4ae482bb7e7cdd28f49c024b047d55fa7c575504a697"
POOL_SHAPE = (16, 32, 32)
PHYSICAL_SCALE = 4
MODEL_SHAPE = tuple(value * PHYSICAL_SCALE for value in POOL_SHAPE)
MATCH_RADIUS_MODEL = 10.0
MAX_FOREGROUND_POINTS = 500_000


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
    os.replace(temporary, path)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    low, high = np.quantile(values, (0.02, 0.998))
    if not np.isfinite(low + high) or high <= low:
        raise ValueError("volume has no finite percentile range")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _virtual_crop(
    volume: np.ndarray, start: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    start = np.asarray(start, dtype=np.int64)
    source_shape = np.asarray(volume.shape, dtype=np.int64)
    target_shape = np.asarray(shape, dtype=np.int64)
    before = np.maximum(-start, 0)
    after = np.maximum(start + target_shape - source_shape, 0)
    padded = np.pad(
        volume,
        tuple((int(a), int(b)) for a, b in zip(before, after)),
        mode="reflect",
    )
    adjusted = start + before
    slices = tuple(
        slice(int(origin), int(origin + width))
        for origin, width in zip(adjusted, target_shape)
    )
    result = np.ascontiguousarray(padded[slices])
    if result.shape != shape:
        raise RuntimeError(f"crop shape mismatch: {result.shape} != {shape}")
    return result


def physical_patch(
    volume: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Restore pooled 1.625-um voxels to the model's fine physical scale."""

    point = np.asarray(point, dtype=np.float32).reshape(3)
    start = np.floor(point).astype(np.int64) - np.asarray(POOL_SHAPE) // 2
    crop = _virtual_crop(normalize_volume(volume), start, POOL_SHAPE)
    tensor = torch.from_numpy(crop)[None, None]
    scaled = F.interpolate(
        tensor,
        size=MODEL_SHAPE,
        mode="trilinear",
        align_corners=False,
    )[0, 0].numpy()
    # align_corners=False maps input p to scale * (p + 0.5) - 0.5.
    target = PHYSICAL_SCALE * (point - start + 0.5) - 0.5
    return np.ascontiguousarray(scaled, dtype=np.float32), target.astype(np.float32)


def local_maximum_probability(
    probability: np.ndarray, point: np.ndarray, radius: float = MATCH_RADIUS_MODEL
) -> float:
    point = np.asarray(point, dtype=np.float32)
    starts = np.maximum(np.floor(point - radius).astype(int), 0)
    stops = np.minimum(
        np.ceil(point + radius + 1).astype(int), np.asarray(probability.shape)
    )
    slices = tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))
    block = probability[slices]
    if not block.size:
        return 0.0
    grid = np.indices(block.shape, dtype=np.float32)
    for axis, origin in enumerate(starts):
        grid[axis] += float(origin)
    distance = np.sqrt(np.sum((grid - point[:, None, None, None]) ** 2, axis=0))
    local = block[distance <= radius]
    return float(local.max()) if local.size else 0.0


def attractor_centroids(
    nucleus_probability: np.ndarray,
    gradient: np.ndarray,
    *,
    foreground_threshold: float = 0.5,
    learning_rate: float = 0.2,
    iterations: int = 200,
    momentum: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode NucVerse3D attractors with the published fixed constants."""

    probability = np.asarray(nucleus_probability, dtype=np.float32)
    vectors = np.asarray(gradient, dtype=np.float32)
    if vectors.shape != probability.shape + (3,):
        raise ValueError("gradient and probability shapes differ")
    starts = np.argwhere(probability >= foreground_threshold)
    original_count = int(len(starts))
    if not original_count:
        return np.empty((0, 3), dtype=np.float32), {
            "foreground_voxels": 0,
            "integrated_voxels": 0,
            "density_threshold": None,
        }
    if len(starts) > MAX_FOREGROUND_POINTS:
        keep = np.linspace(0, len(starts) - 1, MAX_FOREGROUND_POINTS).round().astype(int)
        starts = starts[keep]
    position = starts.astype(np.float64).T
    velocity: float | np.ndarray = 0.0
    for _ in range(iterations):
        sampled = np.stack(
            [
                ndi.map_coordinates(
                    vectors[..., axis], position, order=1, mode="constant", cval=0.0
                )
                for axis in range(3)
            ]
        )
        velocity = learning_rate * sampled + momentum * velocity
        position = position + velocity

    final = position.astype(np.int16)
    bounds = np.asarray(probability.shape, dtype=np.int64)[:, None]
    valid = np.logical_and(final >= 0, final < bounds).all(axis=0)
    final = final[:, valid]
    density = np.zeros(probability.shape, dtype=np.int32)
    if final.shape[1]:
        unique, counts = np.unique(final, axis=1, return_counts=True)
        density[tuple(unique)] = counts.astype(np.int32)
    smoothed = ndi.gaussian_filter(density, 1.0)
    threshold = float(smoothed.mean() + 5.0 * smoothed.std())
    labels, count = ndi.label(smoothed > threshold)
    if count:
        centers = np.asarray(
            ndi.center_of_mass(
                smoothed, labels=labels, index=np.arange(1, count + 1)
            ),
            dtype=np.float32,
        ).reshape(-1, 3)
    else:
        centers = np.empty((0, 3), dtype=np.float32)
    return centers, {
        "foreground_voxels": original_count,
        "integrated_voxels": int(len(starts)),
        "valid_final_voxels": int(final.shape[1]),
        "density_threshold": threshold,
        "centroids": int(len(centers)),
    }


def stable_subset(
    rows: Iterable[dict[str, Any]], *, per_embryo: int
) -> list[dict[str, Any]]:
    by_embryo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_embryo.setdefault(str(row["embryo"]), []).append(row)
    if set(by_embryo) != {"44b6", "6bba"}:
        raise ValueError("screen requires both embryo prefixes")
    selected: list[dict[str, Any]] = []
    for embryo in sorted(by_embryo):
        ranked = sorted(
            by_embryo[embryo],
            key=lambda row: hashlib.sha256(
                f"{RUN_ID}:{row['path']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < per_embryo:
            raise ValueError(f"not enough {embryo} optimization examples")
        selected.extend(ranked[:per_embryo])
    return selected


def load_rows(
    manifest_path: Path, *, phase: str, per_embryo: int
) -> list[dict[str, Any]]:
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("real localization manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("competition_train_data_read") is True
        and manifest.get("competition_test_data_read") is False
        and manifest.get("public_leaderboard_used_for_selection") is False
    ):
        raise ValueError("manifest violates clean screen policy")
    rows = [row for row in manifest["files"] if row["role"] == phase]
    if phase == "optimization":
        return stable_subset(rows, per_embryo=per_embryo)
    if phase == "selection":
        return sorted(rows, key=lambda row: str(row["path"]))
    raise ValueError("only optimization and gated selection phases are supported")


def validate_optimization_receipt(path: Path, onnx_sha256: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not (
        receipt.get("run_id") == RUN_ID
        and receipt.get("phase") == "optimization"
        and receipt.get("compatibility_passed") is True
        and receipt.get("onnx_sha256") == onnx_sha256
        and receipt.get("public_leaderboard_used_for_selection") is False
        and receipt.get("competition_test_data_read") is False
    ):
        raise ValueError("optimization receipt cannot authorize selection")
    return receipt


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    if sha256_file(path) != ONNX_SHA256:
        raise ValueError("converted NucVerse3D ONNX hash changed")
    import onnx
    from onnx2torch import convert

    model = convert(onnx.load(str(path))).eval().to(device)
    return model


@torch.inference_mode()
def predict_patch(
    model: torch.nn.Module, patch: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(patch)[None, ..., None].to(device)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with context:
        outputs = model(tensor)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise RuntimeError("converted model returned unexpected outputs")
    mask, gradient = (value.float().cpu().numpy()[0] for value in outputs)
    if mask.shape != MODEL_SHAPE + (2,) or gradient.shape != MODEL_SHAPE + (3,):
        raise RuntimeError("converted model output shape changed")
    return mask[..., 1], gradient


def screen_row(
    model: torch.nn.Module,
    path: Path,
    *,
    device: torch.device,
    maximum_points: int,
) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        volumes = np.asarray(archive["volumes"])
        nodes = np.asarray(archive["nodes"])
    if volumes.shape != (3, 64, 64, 64):
        raise ValueError(f"unexpected replay volume shape in {path}")
    points = nodes[nodes[:, 0].astype(np.int64) == 1, 1:4].astype(np.float32)
    order = sorted(
        range(len(points)),
        key=lambda index: hashlib.sha256(
            f"{RUN_ID}:{path.name}:{index}".encode("utf-8")
        ).hexdigest(),
    )[:maximum_points]
    results = []
    for point_index in order:
        patch, target = physical_patch(volumes[1], points[point_index])
        probability, gradient = predict_patch(model, patch, device)
        support = local_maximum_probability(probability, target)
        centers, diagnostics = attractor_centroids(probability, gradient)
        if len(centers):
            distance = float(np.linalg.norm(centers - target, axis=1).min())
        else:
            distance = math.inf
        results.append(
            {
                "point_index": int(point_index),
                "target_model_zyx": target.tolist(),
                "maximum_local_probability": support,
                "foreground_supported": support >= 0.5,
                "nearest_attractor_distance_model_voxels": distance,
                "attractor_matched": distance <= MATCH_RADIUS_MODEL,
                "decoder": diagnostics,
            }
        )
    return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = [point for row in rows for point in row["points"]]
    finite = [
        float(point["nearest_attractor_distance_model_voxels"])
        for point in points
        if math.isfinite(float(point["nearest_attractor_distance_model_voxels"]))
    ]
    supported = sum(bool(point["foreground_supported"]) for point in points)
    matched = sum(bool(point["attractor_matched"]) for point in points)
    count = len(points)
    if not count:
        raise ValueError("screen produced no annotated points")
    return {
        "examples": len(rows),
        "points": count,
        "foreground_support_recall": supported / count,
        "attractor_recall": matched / count,
        "finite_attractor_fraction": len(finite) / count,
        "mean_finite_attractor_distance_model_voxels": (
            float(np.mean(finite)) if finite else None
        ),
        "p90_finite_attractor_distance_model_voxels": (
            float(np.quantile(finite, 0.9)) if finite else None
        ),
    }


def compatibility_passed(summary: dict[str, Any]) -> bool:
    mean_distance = summary["mean_finite_attractor_distance_model_voxels"]
    return bool(
        summary["points"] >= 16
        and summary["foreground_support_recall"] >= 0.85
        and summary["attractor_recall"] >= 0.80
        and summary["finite_attractor_fraction"] >= 0.90
        and mean_distance is not None
        and mean_distance <= 8.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("optimization", "selection"), default="optimization")
    parser.add_argument("--optimization-receipt", type=Path)
    parser.add_argument("--per-embryo", type=int, default=8)
    parser.add_argument("--maximum-points-per-example", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.per_embryo < 1 or args.maximum_points_per_example < 1:
        raise ValueError("screen bounds must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    onnx_sha256 = sha256_file(args.onnx)
    if args.phase == "selection":
        if args.optimization_receipt is None:
            raise ValueError("selection requires an optimization receipt")
        validate_optimization_receipt(args.optimization_receipt, onnx_sha256)
    elif args.optimization_receipt is not None:
        raise ValueError("optimization must not consume a prior receipt")

    selected = load_rows(
        args.manifest, phase=args.phase, per_embryo=args.per_embryo
    )
    model = load_model(args.onnx, device)
    rows = []
    for index, source in enumerate(selected, start=1):
        path = args.real_root / str(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"real replay shard changed: {path}")
        points = screen_row(
            model,
            path,
            device=device,
            maximum_points=args.maximum_points_per_example,
        )
        rows.append(
            {
                "path": str(source["path"]),
                "embryo": str(source["embryo"]),
                "sha256": str(source["sha256"]),
                "points": points,
            }
        )
        print(f"SCREEN {index}/{len(selected)} {source['path']} points={len(points)}", flush=True)

    summary = summarize(rows)
    passed = compatibility_passed(summary)
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": args.phase,
        "status": "complete",
        "compatibility_passed": passed,
        "summary": summary,
        "rows": rows,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "onnx_sha256": onnx_sha256,
        "manifest_sha256": sha256_file(args.manifest),
        "model_parameters": 40_458_005,
        "preprocessing": "percentile_2_99.8_then_physical_4x_trilinear",
        "decoder": "published_argmax_gradient_attractor_constants",
        "competition_train_data_read": True,
        "competition_test_data_read": False,
        "public_leaderboard_used_for_selection": False,
        "submission_created": False,
        "authorized_for_submission": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"compatibility_passed": passed, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()

