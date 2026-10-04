"""Private authorized Kaggle upload with bounded resumable chunks, no URL logging."""
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlsplit

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadResult

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha


def bounded_upload(self, path, url, quiet, resume=False):
    if urlsplit(url).hostname not in ('www.googleapis.com','storage.googleapis.com'):
        raise ValueError('Unexpected official blob upload host')
    path=Path(path); size=path.stat().st_size
    session=requests.Session()
    def status():
        response=session.put(url,data=b'',headers={'Content-Length':'0','Content-Range':f'bytes */{size}'},
                             timeout=(10,45),allow_redirects=False)
        if response.status_code in (200,201): return size
        if response.status_code != 308:
            raise RuntimeError(f'Upload status HTTP{response.status_code}')
        value=response.headers.get('Range')
        if value is None: return 0
        match=re.fullmatch(r'bytes=0-(\d+)',value)
        if not match: raise ValueError('Invalid server upload range')
        offset=int(match.group(1))+1
        if not 0 <= offset <= size: raise ValueError('Invalid upload offset')
        return offset
    offset=status() if resume else 0
    start=time.monotonic()
    with path.open('rb') as stream:
        while offset < size:
            if time.monotonic()-start > 900: raise TimeoutError('Blob upload budget exhausted')
            stream.seek(offset); data=stream.read(min(8*1024**2,size-offset))
            stop=offset+len(data)-1
            try:
                response=session.put(url,data=data,headers={
                    'Content-Length':str(len(data)), 'Content-Range':f'bytes {offset}-{stop}/{size}'},
                    timeout=(10,90),allow_redirects=False)
            except requests.RequestException as error:
                print(json.dumps(dict(event='upload_retry',error_type=type(error).__name__)),flush=True)
                offset=status(); continue
            if response.status_code in (200,201):
                if stop != size-1: raise ValueError('Premature upload completion')
                offset=size
            elif response.status_code == 308:
                value=response.headers.get('Range','')
                if value != f'bytes=0-{stop}': raise ValueError('Unexpected acknowledged upload range')
                offset=stop+1
            elif response.status_code in (429,500,502,503,504):
                offset=status(); continue
            else:
                raise RuntimeError(f'Blob upload HTTP{response.status_code}')
            print(json.dumps(dict(event='upload_progress',bytes=offset,total_bytes=size,
                                  elapsed_seconds=time.monotonic()-start)),flush=True)
    return ResumableUploadResult.COMPLETE


def main():
    folder=ROOT/'.biohub/staging/biohub-trajectory-motion-runtime-upload-v2'
    build=json.loads((ROOT/'reports/experiments/trajectory-kaggle-runtime-v2-build.json').read_text())
    if sha(folder/'trajectory-runtime-v1.zip')!=build['archive_sha256']:
        raise ValueError('Frozen revision2 archive changed')
    meta=json.loads((folder/'dataset-metadata.json').read_text())
    if meta['id']!='indarkarhana/biohub-trajectory-motion-runtime-v1' or meta['isPrivate'] is not True:
        raise ValueError('Unexpected upload target or visibility')
    api=KaggleApi(); api.authenticate()
    # Instance-only override, no installed SDK or shared package changes.
    import types
    api.upload_complete=types.MethodType(bounded_upload,api)
    started=time.monotonic()
    timer=threading.Timer(1200,lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        os.chdir(folder)
        response=api.dataset_create_new('.',public=False,quiet=True,convert_to_csv=False)
        payload=response.to_dict()
        # Kaggle API dataset response is nonsecret; whitelist result fields only.
        result=dict(status='api_response',response={k:v for k,v in payload.items()
                    if k in ('status','error','url','datasetId','datasetSlug','ref')},
                    archive_sha256=build['archive_sha256'],private=True,
                    elapsed_seconds=time.monotonic()-started)
        (ROOT/'reports/experiments/trajectory-runtime-v2-upload.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        print(json.dumps(dict(status='failed',error_type=type(error).__name__)),flush=True)
        raise RuntimeError('Private upload failed; signed URLs suppressed') from None
    finally:
        timer.cancel()


if __name__=='__main__': main()
