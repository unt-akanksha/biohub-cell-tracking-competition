"""Observe one existing Kaggle run, then download and verify it exactly once."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
KERNEL='indarkarhana/biohub-event-anchor-acceptance'


def main():
    receipt=ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-watch.json'
    outputs=ROOT/'.biohub/cache/trajectory-event-anchor-kaggle-v1-output'
    assert not receipt.exists() and not outputs.exists(),'Inspect prior watcher/download; never restart blindly'
    launch=json.loads((ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-launch.json').read_text())
    assert launch['kernel']==KERNEL and launch['kernel_pushed'] and launch['exit_code']==0
    cli=Path(sys.executable).parent/'Scripts/kaggle.exe'
    assert cli.exists(),'Run with base Python'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    report=dict(status='observing_existing_run',kernel=KERNEL,kernel_version=1,
                observer_pid=os.getpid(),kernel_launched=False,submission_performed=False,
                started_utc=datetime.now(timezone.utc).isoformat())
    def persist(**values):
        report.update(values,updated_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def run(command,timeout=45):
        return subprocess.run(command,capture_output=True,text=True,encoding='utf-8',
                              env=env,timeout=timeout,check=True,cwd=ROOT)
    persist()
    try:
        deadline=time.monotonic()+5400
        while time.monotonic()<deadline:
            try:
                status=run([str(cli),'kernels','status',KERNEL]).stdout.strip()
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                # Failed observation is never treated as a failed GPU job.
                persist(status='observation_retry',observation_error=type(error).__name__)
                time.sleep(60)
                continue
            match=re.fullmatch(re.escape(KERNEL)+r' has status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"',status)
            assert match,'Unrecognized kernel state: '+status
            state=match.group(1).upper()
            persist(status='observing_existing_run',observed_kernel_status=state)
            if state in ('COMPLETE','ERROR','CANCELLED','CANCELED'):
                break
            assert state in ('RUNNING','QUEUED'),'Unknown nonterminal state: '+state
            time.sleep(60)
        else:
            persist(status='observation_deadline_reached',job_terminal_unproven=True)
            return
        actual=json.loads(run([sys.executable,'scripts/get-kaggle-kernel-state.py','--kernel-slug',KERNEL]).stdout)
        assert actual['present'] and actual['current_version_number']==1
        assert actual['is_private'] and actual['enable_gpu'] and not actual['enable_tpu'] and not actual['enable_internet']
        persist(status='downloading_terminal_outputs',remote_state=actual)
        outputs.mkdir(exist_ok=False)
        downloaded=run([str(cli),'kernels','output',KERNEL,'-p',str(outputs)],timeout=600)
        (outputs/'download-cli.log').write_text(downloaded.stdout+downloaded.stderr,encoding='utf-8')
        persist(status='terminal_outputs_downloaded',outputs=str(outputs))
        if state!='COMPLETE':
            persist(status='kernel_terminal_failure_outputs_preserved')
            return
        analysis=ROOT/'.biohub/cache/graph-analysis-venv/Scripts/python.exe'
        verified=run([str(analysis),'scripts/verify-trajectory-event-anchor-kaggle-v1.py','--outputs',str(outputs)],timeout=600)
        result=json.loads((ROOT/'reports/experiments/trajectory-event-anchor-kaggle-v1-result.json').read_text())
        persist(status='verification_finished',verification_status=result['status'],verification_stdout=verified.stdout)
    except BaseException as error:
        details=dict(status='observer_needs_inspection',error=repr(error),gpu_job_restarted=False)
        if isinstance(error,subprocess.CalledProcessError):
            details.update(stdout=error.stdout,stderr=error.stderr)
        persist(**details)
        raise
    finally:
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
