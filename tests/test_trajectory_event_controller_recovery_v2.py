from pathlib import Path
import runpy
import subprocess
import pytest

retry=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/resume-trajectory-event-source-controller-v2.py'))['observe_with_retry']


def test_connection_retry_observes_same_job_without_relaunch():
    seen=[];pauses=[];job={'remote':{'pid':65151}}
    def observer(value):
        seen.append(value)
        if len(seen)<3:raise subprocess.CalledProcessError(255,['ssh'])
        return {'status':'running'}
    assert retry(observer,job,pause=pauses.append)=={'status':'running'}
    assert len(seen)==3 and all(item is job for item in seen) and pauses==[10,10]


def test_remote_integrity_failure_is_not_retried():
    def observer(value):raise subprocess.CalledProcessError(1,['ssh'])
    with pytest.raises(subprocess.CalledProcessError):retry(observer,{'remote':{'pid':1}},pause=lambda _:pytest.fail('must not retry'))
