"""Fetch only ZIP metadata; save signed URL privately for four movie ranges."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.kaggle_archive_ranges import RangeReader

STEMS = ('44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132')


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
    output = ROOT / '.biohub/cache/trajectory-division-archive-plan-v1'
    if output.exists():
        raise ValueError('Preserve private archive plan')
    start = time.monotonic()
    timer = threading.Timer(300, lambda: os._exit(124)); timer.daemon = True; timer.start()
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        http = client.http_client(); request = ApiDownloadDataFilesRequest()
        request.competition_name = 'biohub-cell-tracking-during-development'
        prepared = http._prepare_request('competitions.CompetitionApiService', 'DownloadDataFiles', request)
        response = http._session.send(prepared, allow_redirects=False, timeout=(10, 20))
        url = response.headers.get('Location', ''); status = response.status_code; response.close()
        if status not in (302, 303, 307) or urlparse(url).hostname != 'storage.googleapis.com':
            raise ValueError('Authorized archive redirect unavailable')
    session = requests.Session()
    head = session.head(url, timeout=(10, 20)); head.raise_for_status()
    length = int(head.headers['Content-Length']); etag = head.headers['ETag']; head.close()
    fetched = []
    def fetch(first, last):
        if not (0 <= first <= last < length) or last - first + 1 > 32 * 1024 ** 2:
            raise ValueError('Invalid archive range')
        response = session.get(url, headers={'Range': f'bytes={first}-{last}',
                               'If-Match': etag, 'Accept-Encoding': 'identity'}, timeout=(10, 30))
        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{length}':
            response.close(); raise ValueError('Server did not honor pinned range')
        data = response.content; response.close()
        if len(data) != last - first + 1:
            raise ValueError('Incomplete ZIP metadata range')
        fetched.append(dict(first=first, last=last, sha256=hashlib.sha256(data).hexdigest()))
        return data
    names = []
    for stem in STEMS:
        base = f'train/{stem}.zarr'
        names.extend((base + '/zarr.json', base + '/0/zarr.json'))
        names.extend(f'{base}/0/c/{t}/0/0/0' for t in range(100))
    with zipfile.ZipFile(RangeReader(length, fetch)) as archive:
        records = []
        for name in names:
            info = archive.getinfo(name)
            if info.is_dir() or info.flag_bits & 1 or info.compress_type not in (0, 8):
                raise ValueError('Invalid authorized movie member')
            records.append(dict(path=name, bytes=info.file_size, compressed_bytes=info.compress_size,
                                crc32=info.CRC, header_offset=info.header_offset, compress_type=info.compress_type))
    total = sum(r['bytes'] for r in records)
    if total > 3 * 1024 ** 3 or len(records) != 408:
        raise ValueError('Four-movie size/inventory cap exceeded')
    output.mkdir(parents=True)
    plan = dict(run_id='trajectory-division-archive-plan-v1', archive_url=url,
                archive_bytes=length, archive_etag=etag, stems=list(STEMS), records=records,
                total_bytes=total, metadata_range_receipts=fetched,
                ground_truth_included=False, authorized_for_submission=False,
                prepared_epoch=time.time(), elapsed_seconds=time.monotonic()-start)
    path = output / 'PRIVATE_ARCHIVE_PLAN.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    timer.cancel(); session.close()
    print(json.dumps(dict(status='prepared', files=len(records), image_bytes=total,
        zip_metadata_bytes=sum(r['last']-r['first']+1 for r in fetched),
        private_plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        elapsed_seconds=plan['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # requests exceptions can contain signed URLs; never print their messages.
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1)
