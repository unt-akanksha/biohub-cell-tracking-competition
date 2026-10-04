"""Fetch only the frozen four full image movies from private short-lived URLs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
from urllib.parse import urlparse

STEMS = ("44b6_24264f12", "44b6_81c256f0", "6bba_23af9eeb", "6bba_f1fde7e0")


def validate_plan(plan):
    expected = set()
    for stem in STEMS:
        base = f"train/{stem}.zarr"
        expected.update([base + "/zarr.json", base + "/0/zarr.json"])
        expected.update(f"{base}/0/c/{t}/0/0/0" for t in range(100))
    records = plan.get("records", [])
    actual = [r["path"] for r in records]
    if (plan.get("run_id") != "public-d4-movie-download-v1" or plan.get("stems") != list(STEMS)
            or plan.get("ground_truth_included") is not False or plan.get("authorized_for_submission") is not False
            or set(actual) != expected or len(actual) != len(expected)):
        raise ValueError("Exact four image-only complete-movie inventory required")
    for row in records:
        path = PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts or urlparse(row["url"]).hostname != "storage.googleapis.com":
            raise ValueError("Untrusted download path or host")
        if not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ValueError("Invalid declared content length")
    if plan["total_bytes"] != sum(r["bytes"] for r in records) or plan["total_bytes"] > 3 * 1024**3:
        raise ValueError("Invalid download size accounting")


def main(args):
    import requests
    payload = args.plan.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.plan_sha256:
        raise ValueError("Private download plan changed")
    plan = json.loads(payload)
    validate_plan(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free - plan["total_bytes"] < 1024**3:
        raise ValueError("Insufficient disk to leave 1 GiB free; no files downloaded")
    started = time.monotonic()
    stop = threading.Event()

    def fetch(row):
        target = args.output / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        md5, sha = hashlib.md5(), hashlib.sha256()
        try:
            with requests.get(row["url"], stream=True, timeout=(15, 45)) as response:
                if response.status_code != 200:
                    raise ValueError(f"HTTP status {response.status_code}")
                if urlparse(response.url).hostname != "storage.googleapis.com":
                    raise ValueError("Unexpected redirect")
                length = 0
                with target.open("xb") as stream:
                    for chunk in response.iter_content(1024**2):
                        if stop.is_set() or time.monotonic()-started > 900:
                            raise ValueError("Data download stopped or time cap exceeded")
                        length += len(chunk)
                        if length > row["bytes"]:
                            raise ValueError("Content larger than declared")
                        sha.update(chunk)
                        md5.update(chunk)
                        stream.write(chunk)
                if length != row["bytes"]:
                    raise ValueError("Incomplete image file")
                declared = dict(item.strip().split("=", 1) for item in row["goog_hash"].split(",") if "=" in item)
                if "md5" in declared and base64.b64encode(md5.digest()).decode() != declared["md5"]:
                    raise ValueError("GCS image checksum mismatch")
                return dict(path=row["path"], bytes=length, sha256=sha.hexdigest())
        except Exception as error:
            stop.set()
            # Do not print a requests exception containing its signed URL.
            raise RuntimeError(f"Image fetch failed for {row['path']}: {type(error).__name__}") from None

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch, plan["records"]):
            records.append(row)
            if len(records) % 50 == 0:
                print(json.dumps(dict(downloaded=len(records), total=len(plan["records"]))), flush=True)
    for stem in STEMS:
        root = args.output / "train" / (stem + ".zarr")
        group = json.loads((root / "zarr.json").read_text())
        array = json.loads((root / "0/zarr.json").read_text())
        if (array["shape"] != [100, 64, 256, 256] or array["data_type"] != "uint16"
                or array["chunk_grid"]["configuration"]["chunk_shape"] != [1, 64, 256, 256]
                or group["node_type"] != "group" or group["zarr_format"] != 3):
            raise ValueError("Image geometry differs from complete-movie contract")
    result = dict(status="complete", run_id="public-d4-movie-download-v1", records=records,
                  stems=list(STEMS), bytes=sum(r["bytes"] for r in records),
                  elapsed_seconds=time.monotonic()-started, ground_truth_opened=False,
                  private_urls_included=False, authorized_for_submission=False)
    (args.output / "IMAGE_MANIFEST.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
