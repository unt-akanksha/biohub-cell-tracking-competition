"""Paced download of trajectory evidence only, excluding installed libraries."""
from datetime import datetime,timezone
import json
from pathlib import Path,PurePosixPath
import time


def evidence_name(name):
    if not name or '\\' in name or ':' in name:return False
    path=PurePosixPath(name)
    if path.is_absolute() or any(p in ('.','..') for p in name.split('/')):return False
    return name=='submission.csv' or (len(path.parts)>1 and path.parts[0] in ('trajectory-complete','trajectory-smoke'))


def required_files(outputs):
    required={'submission.csv','trajectory-complete/submission.csv'}
    for folder,mode in (('trajectory-complete','production'),('trajectory-smoke','smoke')):
        name=folder+'/result.json';required.add(name);path=outputs/name
        if not path.exists():return None
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['status']!='complete' or result['mode']!=mode:raise ValueError('Wrong terminal evidence mode/status')
        for shard,worker in result['workers'].items():
            for stem in worker['movies']:
                prefix=folder+'/shard-'+shard+'/'+stem+'-original/'
                required.update(prefix+n for n in ('pre-postprocess.json','prediction.json','motion-repaired-prediction.json',
                    'structured-repaired-prediction.json','repaired-prediction.json','repair-details.json','raw-candidates.npz'))
    return required


def download(kernel,outputs,receipt,verify_remote):
    import requests
    from kaggle import api
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    from research.trajectory_runtime_v1 import sha
    outputs,receipt=Path(outputs),Path(receipt)
    if outputs.exists() or receipt.exists():raise ValueError('Inspect prior evidence download; do not blindly restart')
    verify_remote();outputs.mkdir()
    report=dict(status='downloading_evidence',kernel=kernel,pages=0,files={},gpu_job_restarted=False)
    def persist(**values):
        report.update(values,updated_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    persist();deadline=time.monotonic()+1200;token=None;owner,slug=kernel.split('/')
    try:
        with api.build_kaggle_client() as client:
            while time.monotonic()<deadline:
                request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug;request.page_size=200
                if token:request.page_token=token
                response=None
                for attempt in range(4):
                    try:
                        response=client.kernels.kernels_api_client.list_kernel_session_output(request);break
                    except requests.HTTPError as error:
                        status=error.response.status_code
                        if status!=429 and status<500:raise RuntimeError('Listing HTTP '+str(status)) from None
                        header=error.response.headers.get('Retry-After','60');delay=max(60,int(header)) if header.isdigit() else 60
                        if time.monotonic()+delay>=deadline:raise RuntimeError('Listing deadline reached') from None
                        persist(status='listing_backoff',http_status=status,attempt=attempt+1);time.sleep(delay)
                if response is None:raise RuntimeError('Listing retries exhausted')
                report['pages']+=1
                if response.log and report['pages']==1:
                    (outputs/(slug+'.log')).write_text(response.log,encoding='utf-8')
                for item in response.files or []:
                    name=item.file_name
                    if not evidence_name(name):continue
                    dest=(outputs/name).resolve()
                    if not dest.is_relative_to(outputs.resolve()):raise ValueError('Unsafe evidence path')
                    try:
                        fetched=requests.get(item.url,timeout=60)
                    except requests.RequestException:raise RuntimeError('Evidence download transport error') from None
                    if fetched.status_code!=200:raise RuntimeError('Evidence download HTTP '+str(fetched.status_code))
                    payload=fetched.content
                    if dest.exists():
                        if dest.read_bytes()!=payload:raise ValueError('Conflicting evidence artifact')
                    else:
                        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
                    report['files'][name]=dict(bytes=len(payload),sha256=sha(dest))
                persist(status='downloading_evidence')
                required=required_files(outputs)
                if required is not None and required<=report['files'].keys():break
                token=response.next_page_token
                if not token:raise RuntimeError('Required evidence absent from terminal outputs')
                time.sleep(1)
            else:raise RuntimeError('Evidence download deadline reached')
        verify_remote();persist(status='required_production_evidence_downloaded')
        return report
    except BaseException as error:
        persist(status='download_needs_inspection',error_type=type(error).__name__);raise
