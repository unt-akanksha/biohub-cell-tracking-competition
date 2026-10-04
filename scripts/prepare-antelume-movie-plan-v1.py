#!/usr/bin/env python
"""Prepare a private, short-lived image download plan for an arbitrary stem set.

This generalises `prepare-public-d4-movie-download-v1.py`, which is hard-pinned to
the four 2026-09-10 diagnostic movies. The selection work needs many more movies
than four, fetched in batches so that peak disk stays small.

Images only. No ground truth is requested here: the GEFF labels are already
cached locally and travel with the bundle instead. The plan carries signed URLs
that expire, so it is generated per batch, immediately before use, and is never
committed.

Read-only against Kaggle. Produces a plan; downloads nothing.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCAL = threading.local()

FRAMES = 100
MAX_BYTES_PER_MOVIE = 1_200 * 1024**2  # generous ceiling; observed ~430 MB


def fetch_url(name: str) -> dict:
    from kaggle.api.kaggle_api_extended import ApiDownloadDataFileRequest, KaggleApi

    if not hasattr(LOCAL, "api"):
        LOCAL.api = KaggleApi()
        LOCAL.api.authenticate()
    with LOCAL.api.build_kaggle_client() as client:
        request = ApiDownloadDataFileRequest()
        request.competition_name = "biohub-cell-tracking-during-development"
        request.file_name = name
        response = client.competitions.competition_api_client.download_data_file(request)
        try:
            url = response.request.url
            length = int(response.headers["Content-Length"])
            if (
                response.status_code != 200
                or urlparse(url).hostname != "storage.googleapis.com"
                or length <= 0
            ):
                raise ValueError(f"Unexpected download response for {name}")
            return {
                "path": name,
                "url": url,
                "bytes": length,
                "goog_hash": response.headers.get("x-goog-hash", ""),
            }
        finally:
            response.close()


def file_names(stems) -> list[str]:
    names: list[str] = []
    for stem in stems:
        base = f"train/{stem}.zarr"
        names.append(base + "/zarr.json")
        names.append(base + "/0/zarr.json")
        names.extend(f"{base}/0/c/{t}/0/0/0" for t in range(FRAMES))
    return names


def validate_plan(plan: dict) -> None:
    """Re-derive the expected inventory and refuse anything else."""
    expected = set(file_names(plan["stems"]))
    records = plan.get("records", [])
    actual = [row["path"] for row in records]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Plan inventory does not match the requested stems exactly")
    if plan.get("ground_truth_included") is not False:
        raise ValueError("Image-only plan required")
    for row in records:
        path = PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Untrusted download path")
        if urlparse(row["url"]).hostname != "storage.googleapis.com":
            raise ValueError("Untrusted download host")
        if not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ValueError("Invalid declared content length")
    ceiling = MAX_BYTES_PER_MOVIE * len(plan["stems"])
    if plan["total_bytes"] != sum(r["bytes"] for r in records) or plan["total_bytes"] > ceiling:
        raise ValueError("Invalid download size accounting")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems", required=True, help="comma-separated, or @file with one per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.stems.startswith("@"):
        stems = [s.strip() for s in Path(args.stems[1:]).read_text().split() if s.strip()]
    else:
        stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    if not stems or len(set(stems)) != len(stems):
        raise SystemExit("A non-empty set of distinct stems is required")

    destination = Path(args.output)
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite an existing plan: {destination}")

    names = file_names(stems)
    started = time.monotonic()
    records: list[dict] = []
    # Modest parallelism: read-only, and a failure should stop rather than hammer Kaggle.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(fetch_url, names):
            records.append(record)
            if len(records) % 100 == 0:
                print(f"  prepared {len(records)}/{len(names)} urls", flush=True)

    plan = {
        "run_id": "antelume-config-sweep-v1",
        "stems": list(stems),
        "records": records,
        "total_bytes": sum(row["bytes"] for row in records),
        "frames": FRAMES,
        "ground_truth_included": False,
        "authorized_for_submission": False,
        "prepared_epoch": time.time(),
    }
    validate_plan(plan)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan), encoding="utf-8")
    print(
        f"stems {len(stems)} | files {len(records)} | "
        f"{plan['total_bytes'] / 1e9:.2f} GB | {time.monotonic() - started:.1f}s"
    )
    print(f"plan written to {destination}")
    print("NOTE: signed URLs are short-lived; download promptly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
