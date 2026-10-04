"""Paced, smoke-only recovery after output-list rate limiting; no GPU launch."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import requests
from kaggle import api
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.trajectory_runtime_v1 import sha
KERNEL='indarkarhana/biohub-event-anchor-acceptance'


def main():
    receipt=ROOT/'reports/experiments/trajectory-event-anchor-smoke-recovery-v1.json'
    assert not receipt.exists(),'Inspect existing recovery before restarting'
    outputs=ROOT/'.biohub/cache/trajectory-event-anchor-kaggle-v1-output'
    report=dict(status='rate_limit_cooldown',kernel=KERNEL,version=1,pages=0,files={},gpu_job_restarted=False)
    def persist(**values):
        report.update(values,updated_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def required_complete():
        path=outputs/'trajectory-smoke/result.json'
        if not path.exists():return False
        result=json.loads(path.read_text());assert result['status']=='complete' and result['mode']=='smoke'
        needed={'trajectory-smoke/result.json'}
        for shard,worker in result['workers'].items():
            for stem in worker['movies']:
                prefix='trajectory-smoke/shard-'+shard+'/'+stem+'-original/'
                needed.update(prefix+n for n in ('pre-postprocess.json','structured-repaired-prediction.json','repaired-prediction.json','repair-details.json'))
        return needed<=report['files'].keys()
    persist();time.sleep(60);deadline=time.monotonic()+1200;token=None
    try:
        with api.build_kaggle_client() as client:
            while time.monotonic()<deadline:
                request=ApiListKernelSessionOutputRequest();request.user_name='indarkarhana'
                request.kernel_slug='biohub-event-anchor-acceptance';request.page_size=200
                if token:request.page_token=token
                response=None
                for attempt in range(4):
                    try:
                        response=client.kernels.kernels_api_client.list_kernel_session_output(request)
                        break
                    except requests.HTTPError as error:
                        status=error.response.status_code
                        if status!=429 and status<500:raise RuntimeError('Listing failed HTTP '+str(status)) from None
                        retry=error.response.headers.get('Retry-After','60')
                        delay=max(60,int(retry)) if retry.isdigit() else 60
                        if time.monotonic()+delay>=deadline:raise RuntimeError('Listing recovery deadline reached') from None
                        persist(status='listing_backoff',http_status=status,attempt=attempt+1)
                        time.sleep(delay)
                if response is None:raise RuntimeError('Listing retries exhausted; no GPU rerun')
                report['pages']+=1
                for item in response.files or []:
                    name=item.file_name
                    if not name.startswith('trajectory-smoke/'):continue
                    dest=(outputs/name).resolve()
                    if not dest.is_relative_to((outputs/'trajectory-smoke').resolve()):raise ValueError('Unsafe smoke output path')
                    try:
                        downloaded=requests.get(item.url,timeout=60)
                        if downloaded.status_code!=200:raise RuntimeError('Smoke download failed HTTP '+str(downloaded.status_code))
                    except requests.RequestException:
                        raise RuntimeError('Smoke download transport error; inspect recovery receipt') from None
                    payload=downloaded.content
                    if dest.exists():
                        assert dest.read_bytes()==payload,'Existing smoke artifact differs'
                    else:
                        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
                    report['files'][name]=dict(bytes=len(payload),sha256=sha(dest))
                persist(status='listing_and_downloading_smoke')
                if required_complete():break
                token=response.next_page_token
                if not token:raise RuntimeError('Listing ended without required smoke evidence')
                time.sleep(1)
            else:raise RuntimeError('Smoke recovery deadline reached')
        actual=subprocess.run([sys.executable,str(ROOT/'scripts/get-kaggle-kernel-state.py'),'--kernel-slug',KERNEL],
            capture_output=True,text=True,check=True,timeout=45)
        state=json.loads(actual.stdout);assert state['present'] and state['current_version_number']==1
        persist(status='required_smoke_evidence_recovered',remote_version_verified=1)
        print(json.dumps(dict(status=report['status'],pages=report['pages'],files=len(report['files']))),flush=True)
    except BaseException as error:
        persist(status='recovery_needs_inspection',error_type=type(error).__name__)
        raise


if __name__=='__main__':main()
