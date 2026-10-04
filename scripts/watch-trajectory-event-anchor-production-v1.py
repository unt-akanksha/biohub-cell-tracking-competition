"""Observe the existing public test, then harvest only reproducibility evidence."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.kaggle_trajectory_evidence_v1 import download
KERNEL='indarkarhana/biohub-event-anchor-candidate'


def main():
    receipt=ROOT/'reports/experiments/trajectory-event-anchor-production-v1-watch.json'
    assert not receipt.exists(),'Observe the existing watcher, never duplicate it'
    launch=json.loads((ROOT/'reports/experiments/trajectory-event-anchor-production-v1-launch.json').read_text())
    assert launch['kernel']==KERNEL and launch['kernel_pushed'] and launch['exit_code']==0 and launch['public_test_only']
    cli=Path(sys.executable).parent/'Scripts/kaggle.exe';assert cli.exists()
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    report=dict(status='observing_existing_public_test',kernel=KERNEL,version=1,observer_pid=os.getpid(),submission_performed=False)
    def persist(**values):
        report.update(values,updated_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def run(command):return subprocess.run(command,capture_output=True,text=True,env=env,check=True,timeout=45)
    def verify_remote():
        actual=json.loads(run([sys.executable,str(ROOT/'scripts/get-kaggle-kernel-state.py'),'--kernel-slug',KERNEL]).stdout)
        assert actual['present'] and actual['current_version_number']==1 and actual['is_private']
        assert actual['enable_gpu'] and not actual['enable_tpu'] and not actual['enable_internet']
        assert actual['competition_sources']==['biohub-cell-tracking-during-development']
        assert actual['dataset_sources']==['indarkarhana/biohub-trajectory-motion-runtime-v1']
    persist();deadline=time.monotonic()+5400
    try:
        while time.monotonic()<deadline:
            try:status=run([str(cli),'kernels','status',KERNEL]).stdout.strip()
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                persist(status='observation_retry',error_type=type(error).__name__);time.sleep(60);continue
            match=re.fullmatch(re.escape(KERNEL)+r' has status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"',status)
            assert match,'Unrecognized saved-kernel state'
            state=match.group(1).upper();persist(status='observing_existing_public_test',observed_kernel_status=state)
            if state=='COMPLETE':break
            if state in ('ERROR','CANCELLED','CANCELED'):
                persist(status='public_test_terminal_failure');return
            assert state in ('RUNNING','QUEUED');time.sleep(60)
        else:
            persist(status='observation_deadline_reached',job_terminal_unproven=True);return
        persist(status='downloading_terminal_evidence')
        proof=download(KERNEL,ROOT/'.biohub/cache/trajectory-event-anchor-production-v1-output',
            ROOT/'reports/experiments/trajectory-event-anchor-production-v1-harvest.json',verify_remote)
        persist(status='evidence_downloaded_requires_verification',harvest_status=proof['status'])
    except BaseException as error:
        persist(status='observer_needs_inspection',error_type=type(error).__name__);raise
    finally:print(json.dumps(report),flush=True)


if __name__=='__main__':main()
