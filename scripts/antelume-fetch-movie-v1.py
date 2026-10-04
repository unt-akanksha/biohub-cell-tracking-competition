#!/usr/bin/env python
"""Fetch one movie's images directly on Antelume. Runs ON the instance.

Downloading through the laptop failed: the Kaggle client follows its redirect and
pulls the whole body, so preparing signed URLs locally meant moving 1.7 GB over a
slow link before anything could start. The instance has its own credentials and a
fast link, so it fetches its own images.

Images only -- the 102 files that make up one `train/<stem>.zarr`. Ground truth
is never requested here; the GEFF labels travel in the bundle.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FRAMES = 100
LOCAL = threading.local()


def file_names(stem: str) -> list[str]:
    base = f"train/{stem}.zarr"
    names = [base + "/zarr.json", base + "/0/zarr.json"]
    names.extend(f"{base}/0/c/{t}/0/0/0" for t in range(FRAMES))
    return names


def fetch_one(spec) -> tuple[str, int]:
    name, root = spec
    from kaggle.api.kaggle_api_extended import ApiDownloadDataFileRequest, KaggleApi

    if not hasattr(LOCAL, "api"):
        LOCAL.api = KaggleApi()
        LOCAL.api.authenticate()
    target = root / name
    if target.exists() and target.stat().st_size > 0:
        return name, target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL.api.build_kaggle_client() as client:
        request = ApiDownloadDataFileRequest()
        request.competition_name = "biohub-cell-tracking-during-development"
        request.file_name = name
        response = client.competitions.competition_api_client.download_data_file(request)
        try:
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code} for {name}")
            payload = response.content
        finally:
            response.close()
    if not payload:
        raise ValueError(f"Empty payload for {name}")
    target.write_bytes(payload)
    return name, len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--root", required=True, help="image root; files land under <root>/train/...")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    names = file_names(args.stem)
    started = time.monotonic()
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _name, size in pool.map(fetch_one, [(n, root) for n in names]):
            total += size

    elapsed = time.monotonic() - started
    zarr_dir = root / "train" / f"{args.stem}.zarr"
    present = sum(1 for _ in zarr_dir.rglob("*") if _.is_file())
    if present != len(names):
        raise SystemExit(f"Expected {len(names)} files, found {present}")
    print(json.dumps({
        "event": "movie_fetched", "stem": args.stem, "files": present,
        "bytes": total, "seconds": round(elapsed, 1),
        "mb_per_s": round(total / 1e6 / max(elapsed, 1e-9), 1),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
