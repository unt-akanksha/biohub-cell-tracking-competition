"""Stage exact weights, two already-exposed training frames, and reviewed code."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from public_d4_correction import sha256  # noqa: E402

EXPECTED = {
    "primary.pth": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    "secondary.pth": "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
    "deepcenter.pt": "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0",
    "temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac",
    "simple_node_transformer.py": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
    "train_unet_transformer.py": "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea",
}


def main() -> None:
    import numpy as np
    import zarr

    destination = ROOT / ".biohub/cache/public-d4-preflight-v1"
    if destination.exists():
        raise ValueError("Do not overwrite a staged GPU preflight")
    audit_path = ROOT / "reports/experiments/public-d4-correction-v1-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["geometry"]["passed"] or audit["authorized_for_submission"]:
        raise ValueError("Source-only passing geometry receipt required")
    source_root = ROOT / audit["artifact_root"]
    files = {
        "primary.pth": ROOT / ".biohub/cache/public-d4-preflight-assets-v1/primary/edge_predictor_best.pth",
        "secondary.pth": ROOT / ".biohub/cache/public-d4-preflight-assets-v1/secondary/edge_predictor_best.pth",
        "deepcenter.pt": ROOT / ".biohub/cache/public-d4-preflight-assets-v1/deepcenter/best.pt",
        **{name: ROOT / ".biohub/cache/datasets/biohub-support-source" / name for name in
           ("temporal_unet.py", "simple_node_transformer.py", "train_unet_transformer.py")},
        "public_d4_preflight.py": ROOT / "research/public_d4_preflight.py",
        "run-public-d4-preflight-v1.py": ROOT / "scripts/run-public-d4-preflight-v1.py",
    }
    for name in ("public-predictor-original.py", "public-predictor-d4-corrected.py",
                 "public-postprocess-d4-corrected.py"):
        files[name] = source_root / name
    expected = {**EXPECTED, **{name: row["sha256"] for name, row in audit["artifacts"].items()}}
    hashes = {}
    for name, path in files.items():
        actual = sha256(path.read_bytes())
        if name in expected and actual != expected[name]:
            raise ValueError(f"Public asset/source drift: {name}")
        hashes[name] = dict(sha256=actual, bytes=path.stat().st_size)
    # This movie is a previously exposed public-control training diagnostic,
    # NOT an independent validation movie. Read no GEFF or annotation files.
    movie = ROOT / ".biohub/cache/competition-real-localization-replay-v1/train/44b6_24264f12.zarr"
    frame_ids = [46, 47]
    inputs = {}
    for name in ["zarr.json", "0/zarr.json"] + [f"0/c/{t}/0/0/0" for t in frame_ids]:
        path = movie / name
        if not path.is_file():
            raise ValueError(f"Missing real replay chunk, not a zero-fill frame: {name}")
        inputs[name] = sha256(path.read_bytes())
    group = zarr.open_group(str(movie), mode="r")
    raw = np.stack([group["0"][t] for t in frame_ids])
    if raw.shape != (2, 64, 256, 256) or raw.dtype != np.uint16 or not raw.any():
        raise ValueError("Original full-frame smoke geometry changed")
    quantiles = dict(group.attrs["image_statistics"]["quantiles"])
    destination.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        shutil.copyfile(path, destination / name)
    np.savez_compressed(destination / "frames.npz", raw=raw)
    hashes["frames.npz"] = dict(sha256=sha256((destination / "frames.npz").read_bytes()),
                               bytes=(destination / "frames.npz").stat().st_size)
    manifest = dict(
        run_id="public-d4-preflight-v1", status="staged_not_launched", files=hashes,
        movie="44b6_24264f12", frames=frame_ids, raw_shape=list(raw.shape), quantiles=quantiles,
        original_image_sha256=inputs, geometry_audit_sha256=sha256(audit_path.read_bytes()),
        ground_truth_opened=False, new_target_movies_opened=0, public_predictions_used=False,
        worst_case_seconds=1200, cuda_memory_fraction=0.45, cpu_threads=2,
        models_loaded_sequentially=False, model_training=False,
        competition_submission_performed=False, authorized_for_submission=False,
        data_scope="Two previously exposed training frames, label-free functionality only",
    )
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(destination=str(destination), total_bytes=sum(v["bytes"] for v in hashes.values()),
                         manifest_sha256=sha256((destination / "MANIFEST.json").read_bytes())), indent=2))


if __name__ == "__main__":
    main()
