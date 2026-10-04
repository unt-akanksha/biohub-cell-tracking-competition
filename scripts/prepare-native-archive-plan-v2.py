"""Authorize bounded selected-frame archive reads; never print signed URLs."""
import hashlib
import argparse
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


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
    parser=argparse.ArgumentParser();parser.add_argument('--plan-dir',type=Path,default=ROOT/'.biohub/cache/native-correspondence-v2-plan')
    args=parser.parse_args();root=args.plan_dir
    target = root/'PRIVATE_ARCHIVE_PLAN.json'
    if target.exists():
        raise ValueError('Preserve existing private plan')
    movie_bytes = (root/'MOVIES.json').read_bytes(); movies = json.loads(movie_bytes)
    if movies['competition_test_data_read'] or movies['sealed_audit_opened']:
        raise ValueError('Ineligible movie plan')
    timer = threading.Timer(300, lambda: os._exit(124)); timer.daemon=True; timer.start()
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        http = client.http_client(); request = ApiDownloadDataFilesRequest()
        request.competition_name='biohub-cell-tracking-during-development'
        prepared=http._prepare_request('competitions.CompetitionApiService','DownloadDataFiles',request)
        response=http._session.send(prepared,allow_redirects=False,timeout=(10,20))
        url=response.headers.get('Location',''); status=response.status_code; response.close()
        if status not in (302,303,307) or urlparse(url).hostname != 'storage.googleapis.com':
            raise ValueError('Authorized archive redirect unavailable')
    session=requests.Session(); response=session.head(url,timeout=(10,20)); response.raise_for_status()
    length=int(response.headers['Content-Length']); etag=response.headers['ETag']; response.close()
    def fetch(first,last):
        response=session.get(url,headers={'Range':f'bytes={first}-{last}', 'If-Match':etag,
                             'Accept-Encoding':'identity'},timeout=(10,30))
        try:
            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {first}-{last}/{length}':
                raise ValueError('Pinned archive range not honored')
            data=response.content
            if len(data)!=last-first+1:
                raise ValueError('Incomplete range')
            return data
        finally:
            response.close()
    records=[]
    with zipfile.ZipFile(RangeReader(length,fetch)) as archive:
        for movie in movies['movies']:
            prefix='train/'+movie['stem']+'.zarr/'
            names=[prefix+'zarr.json',prefix+'0/zarr.json']+[prefix+f'0/c/{t}/0/0/0' for t in movie['image_frames']]
            for name in names:
                info=archive.getinfo(name)
                if info.is_dir() or info.flag_bits&1 or info.compress_type not in (0,8):
                    raise ValueError('Unsupported member')
                records.append(dict(path=name,bytes=info.file_size,compressed_bytes=info.compress_size,
                                    crc32=info.CRC,header_offset=info.header_offset,compress_type=info.compress_type))
    image_bytes=sum(r['bytes'] for r in records)
    if image_bytes > 24*1024**3 or len(records)>6000:
        raise ValueError('Selected-frame transfer exceeds declared bounds')
    plan=dict(run_id=movies['run_id'],archive_url=url,archive_bytes=length,archive_etag=etag,
              records=records,total_image_bytes=image_bytes,movie_plan_sha256=hashlib.sha256(movie_bytes).hexdigest(),
              prepared_epoch=time.time(),authorized_for_submission=False)
    target.write_text(json.dumps(plan,indent=2)+'\n'); session.close(); timer.cancel()
    print(json.dumps(dict(status='prepared',files=len(records),image_bytes=image_bytes,
                         private_plan_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed',error_type=type(error).__name__)))
        raise SystemExit(1)
