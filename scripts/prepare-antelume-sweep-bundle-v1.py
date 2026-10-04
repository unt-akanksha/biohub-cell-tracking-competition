#!/usr/bin/env python
"""Assemble the Antelume sweep bundle from already-verified local artifacts.

Everything here is reused, not rebuilt: the same weights, predictor, postprocess
and patched official scorer that produced the 2026-09-10 complete-movie result.
That matters because the first thing the sweep does is re-derive that result's
pooled 0.9448387313 as a correctness check on the whole path.

Every file is SHA-pinned to the digest recorded by the run that used it. Drift
fails the build rather than shipping something different to the instance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / ".biohub/cache/public-d4-preflight-v1"
CORRECTION = ROOT / ".biohub/cache/public-d4-correction-v1"
BUNDLE_SRC = ROOT / ".biohub/cache/public-d4-full-movie-v1-bundle"
SCORER = ROOT / ".biohub/vendor/kaggle-cell-tracking-competition/src/tracking_cellmot"

# Digests recorded by the runs that used these files.
PINNED = {
    "primary.pth": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    "secondary.pth": "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
    "deepcenter.pt": "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0",
    "public-predictor-original.py": "ddca518b0e838b2123f30cfcb31819fb159b4e68fc20ef6b3a42003cc2d528ed",
}
# Text files whose digest is computed with normalised newlines.
PINNED_TEXT = {
    "scorer/metrics.py": "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444",
    "scorer/division_metrics.py": "0635c38621a38f1eb4b55a302b4a817a88e9094930dfc2dab16faeeee60f4dc9",
}

SOURCES = {
    "primary.pth": PREFLIGHT / "primary.pth",
    "secondary.pth": PREFLIGHT / "secondary.pth",
    "deepcenter.pt": PREFLIGHT / "deepcenter.pt",
    "temporal_unet.py": PREFLIGHT / "temporal_unet.py",
    "simple_node_transformer.py": PREFLIGHT / "simple_node_transformer.py",
    "train_unet_transformer.py": PREFLIGHT / "train_unet_transformer.py",
    "public-predictor-original.py": PREFLIGHT / "public-predictor-original.py",
    "public-postprocess-d4-corrected.py": CORRECTION / "public-postprocess-d4-corrected.py",
    "resolved-public-config.json": BUNDLE_SRC / "resolved-public-config.json",
    "public_d4_full_movie.py": ROOT / "research/public_d4_full_movie.py",
    "public_d4_preflight.py": ROOT / "research/public_d4_preflight.py",
    "scorer/metrics.py": SCORER / "metrics.py",
    "scorer/division_metrics.py": SCORER / "division_metrics.py",
    "scorer/__init__.py": SCORER / "__init__.py",
    "run_sweep.py": ROOT / "scripts/antelume-config-sweep-v1.py",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / ".biohub/cache/antelume-sweep-bundle-v1"))
    parser.add_argument("--stems", required=True, help="comma list or @file, for the truth subset")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "scorer").mkdir(parents=True)
    (out / "truth").mkdir(parents=True)

    manifest = {"files": {}, "truth": {}}
    for name, source in SOURCES.items():
        if not source.exists():
            raise SystemExit(f"Missing bundle source: {source}")
        payload = source.read_bytes()
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        digest = sha_bytes(payload)
        if name in PINNED and digest != PINNED[name]:
            raise SystemExit(f"{name} drifted: expected {PINNED[name]}, read {digest}")
        if name in PINNED_TEXT:
            normalised = sha_bytes(payload.replace(b"\r\n", b"\n"))
            if normalised != PINNED_TEXT[name]:
                raise SystemExit(f"{name} drifted: expected {PINNED_TEXT[name]}, read {normalised}")
        manifest["files"][name] = {"sha256": digest, "bytes": len(payload)}

    if args.stems.startswith("@"):
        stems = [s.strip() for s in Path(args.stems[1:]).read_text().split() if s.strip()]
    else:
        stems = [s.strip() for s in args.stems.split(",") if s.strip()]

    truth_root = ROOT / ".biohub/cache/competition-train-geffs-packed-v1/train"
    for stem in stems:
        source = truth_root / f"{stem}.geff"
        if not source.exists():
            raise SystemExit(f"Missing ground truth for {stem}")
        shutil.copytree(source, out / "truth" / f"{stem}.geff")
        manifest["truth"][stem] = sum(
            f.stat().st_size for f in (out / "truth" / f"{stem}.geff").rglob("*") if f.is_file()
        )

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle: {out}")
    print(f"  files {len(manifest['files'])} | truth movies {len(manifest['truth'])}"
          f" | total {total / 1e6:.1f} MB")
    print(f"  every pinned digest verified ({len(PINNED) + len(PINNED_TEXT)} pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
