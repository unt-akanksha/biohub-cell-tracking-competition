"""Download exactly four image movies by verified ZIP member ranges, no GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
from urllib.parse import urlparse
import requests

STEMS = ('44b6_12dfb391', '44b6_267148e4', '6bba_062c8d37', '6bba_07e24132')


def main(args):
    payload = args.plan.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.plan_sha256:
        raise ValueError('Private archive plan changed')
    plan = json.loads(payload)
    if (plan['stems'] != list(STEMS) or plan['ground_truth_included'] is not False
            or urlparse(plan['archive_url']).hostname != 'storage.googleapis.com'
            or plan['authorized_for_submission'] is not False):
        raise ValueError('Invalid image-only plan')
    expected = {f'train/{s}.zarr/{tail}' for s in STEMS
                for tail in ['zarr.json', '0/zarr.json'] + [f'0/c/{t}/0/0/0' for t in range(100)]}
    records = plan['records']
    if len(records) != 408 or {r['path'] for r in records} != expected:
        raise ValueError('Require exact complete four-movie inventory')
    for r in records:
        p = PurePosixPath(r['path'])
        if p.is_absolute() or '..' in p.parts or not 0 < r['bytes'] <= 16 * 1024 ** 2:
            raise ValueError('Unsafe archive member')
    if sum(r['bytes'] for r in records) != plan['total_bytes'] or plan['total_bytes'] > 3 * 1024 ** 3:
        raise ValueError('Invalid declared image size')
    if hashlib.sha256(args.helper.read_bytes()).hexdigest() != args.helper_sha256:
        raise ValueError('Range extractor changed')
    spec = importlib.util.spec_from_file_location('archive_ranges', args.helper)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free - plan['total_bytes'] < 1024 ** 3:
        raise ValueError('Download would leave less than1GiB free')
    started = time.monotonic(); local = threading.local(); stopped = threading.Event()
    completed = []
    def timeout():
        stopped.set()
        (args.output / 'timeout.json').write_text(json.dumps(dict(status='timeout', maximum_seconds=1200)))
        os._exit(124)
    timer = threading.Timer(1200, timeout); timer.daemon = True; timer.start()
    def download(row):
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        def fetch(first, last):
            if stopped.is_set() or not (0 <= first <= last < plan['archive_bytes']) or last-first+1 > 17*1024**2:
                raise ValueError('Invalid or cancelled member range')
            response = local.session.get(plan['archive_url'], stream=True,
                headers={'Range': f'bytes={first}-{last}', 'If-Match': plan['archive_etag'],
                         'Accept-Encoding': 'identity'}, timeout=(10, 30))
            try:
                if response.status_code != 206 or response.headers.get('Content-Range') != f"bytes {first}-{last}/{plan['archive_bytes']}":
                    raise ValueError('Server did not honor pinned byte range')
                data = response.raw.read(last-first+2, decode_content=False)
                if len(data) != last-first+1:
                    raise ValueError('Incomplete or oversized archive range')
                return data
            finally:
                response.close()
        try:
            data = helper.extract_member(row, fetch)
            target = args.output / row['path']; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            return dict(path=row['path'], bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), crc32=row['crc32'])
        except Exception as error:
            stopped.set()
            raise RuntimeError(f"Member failed: {row['path']} ({type(error).__name__})") from None
    # A real chunk/header/decompression/CRC smoke precedes the parallel transfer.
    first = next(r for r in records if r['path'] == f'train/{STEMS[0]}.zarr/0/c/0/0/0/0')
    completed.append(download(first))
    print(json.dumps(dict(event='archive_member_smoke_passed', bytes=completed[0]['bytes'])), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, row) for row in records if row is not first]
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed) % 25 == 0:
                print(json.dumps(dict(event='image_files_downloaded', completed=len(completed), total=408)), flush=True)
    for stem in STEMS:
        root = args.output / 'train' / (stem + '.zarr')
        group = json.loads((root / 'zarr.json').read_text()); array = json.loads((root / '0/zarr.json').read_text())
        if (group['node_type'] != 'group' or group['zarr_format'] != 3
                or array['shape'] != [100, 64, 256, 256] or array['data_type'] != 'uint16'
                or array['chunk_grid']['configuration']['chunk_shape'] != [1, 64, 256, 256]):
            raise ValueError('Image metadata does not match complete inference contract')
    result = dict(status='complete', run_id='trajectory-division-archive-v1',
                  stems=list(STEMS), records=sorted(completed, key=lambda r:r['path']),
                  bytes=sum(r['bytes'] for r in completed), private_plan_sha256=args.plan_sha256,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=args.helper_sha256, elapsed_seconds=time.monotonic()-started,
                  ground_truth_opened=False, private_urls_included=False, authorized_for_submission=False)
    (args.output / 'IMAGE_MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
    timer.cancel()
    print(json.dumps({k:v for k,v in result.items() if k!='records'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('plan', 'helper', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--plan-sha256', required=True); p.add_argument('--helper-sha256', required=True)
    try:
        main(p.parse_args())
    except Exception as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1)
