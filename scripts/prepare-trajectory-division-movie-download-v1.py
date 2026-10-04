"""Create short-lived image-only download URLs for four division-positive validation movies.

URLs are private credential-like artifacts in ignored cache and never printed.
Kaggle credentials themselves are not copied to the cloud instance.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STEMS = ("44b6_12dfb391", "44b6_267148e4", "6bba_062c8d37", "6bba_07e24132")
DESTINATION = ROOT / ".biohub/cache/trajectory-division-movie-download-v1"
LOCAL = threading.local()


def fetch_url(name):
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFileRequest
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
            if response.status_code != 200 or urlparse(url).hostname != "storage.googleapis.com" or length <= 0:
                raise ValueError("Unexpected image download response")
            return dict(path=name, url=url, bytes=length, goog_hash=response.headers.get("x-goog-hash", ""))
        finally:
            response.close()


def main():
    if DESTINATION.exists():
        raise ValueError("Do not overwrite a private download plan")
    names = []
    for stem in STEMS:
        base = f"train/{stem}.zarr"
        names.extend([base + "/zarr.json", base + "/0/zarr.json"])
        names.extend(f"{base}/0/c/{t}/0/0/0" for t in range(100))
    started = time.monotonic()
    # Only four parallel read-only requests; failures stop instead of retrying
    # aggressively against Kaggle. No file body is intentionally consumed.
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(fetch_url, names):
            records.append(record)
            if len(records) % 50 == 0:
                print(json.dumps(dict(prepared_files=len(records), total_files=len(names))), flush=True)
    total = sum(row["bytes"] for row in records)
    if total > 3 * 1024**3:
        raise ValueError("Four-movie download exceeds the declared 3 GiB ceiling")
    DESTINATION.mkdir(parents=True, exist_ok=False)
    plan = dict(run_id="trajectory-division-movie-download-v1", stems=list(STEMS), records=records,
                total_bytes=total, ground_truth_included=False, new_target_movies_opened=0,
                authorized_for_submission=False, prepared_epoch=time.time())
    (DESTINATION / "PRIVATE_DOWNLOAD_PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    print(json.dumps(dict(status="prepared", files=len(records), bytes=total,
                         elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()


