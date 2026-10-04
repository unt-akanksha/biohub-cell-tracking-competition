"""Resume observation of verified batch 3; never restart its GPU process."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def observe_with_retry(observer,launch,pause=time.sleep):
    for attempt in range(3):
        try:return observer(launch)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
            if isinstance(error,subprocess.CalledProcessError) and error.returncode!=255:raise
            if attempt==2:raise
            print(json.dumps(dict(event='observation_connection_retry',attempt=attempt+1,
                                  same_pid=launch['remote']['pid'],gpu_relaunch=False)),flush=True)
            pause(10)


def main():
    assert sys.argv[1:]==['--active-batch','3','--through-batch','7']
    previous=ROOT/'reports/experiments/trajectory-event-source-v1-controller.json'
    old=json.loads(previous.read_text())
    assert old['status']=='stopped_requires_inspection' and old['active_batch']==3
    assert old['remote_root']=='/dev/shm/biohub-event-source-v1-b3.hpf9EX' and old['pid']==65151
    base=ROOT/'scripts/collect-trajectory-event-source-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest()=='3d78bc2fead37777482c8041ebb48008eea4a79f3ddab6d31ea42696a84243d8'
    source=base.read_text(encoding='utf-8')
    old_name='trajectory-event-source-v1-controller.json';new_name='trajectory-event-source-v1-controller-r2.json'
    assert source.count(old_name)==1;source=source.replace(old_name,new_name)
    assert source.count('completed=[]')==1;source=source.replace('completed=[]','completed='+repr(old['completed']))
    marker="model_fitting_or_submission_authorized=False,foreign_jobs_must_not_be_interrupted=True)"
    assert source.count(marker)==1
    source=source.replace(marker,"model_fitting_or_submission_authorized=False,foreign_jobs_must_not_be_interrupted=True,\n                 resumed_observation_only=True,previous_controller_sha256="+repr(hashlib.sha256(previous.read_bytes()).hexdigest())+")")
    namespace=dict(__name__='recovered_controller',__file__=__file__)
    exec(compile(source,str(base),'exec'),namespace)
    observer=namespace['remote_state']
    launch=json.loads((ROOT/'reports/experiments/trajectory-event-source-v1-b3-full-launch.json').read_text())
    assert launch['remote']['pid']==65151 and launch['remote_root']==old['remote_root']
    current=observe_with_retry(observer,launch)
    assert current['status'] in ('running','complete_prelabel_predictions','failed','timeout')
    print(json.dumps(dict(event='existing_job_revalidated',status=current['status'],pid=65151,relaunched=False)),flush=True)
    namespace['remote_state']=lambda job:observe_with_retry(observer,job)
    namespace['main']()


if __name__=='__main__':main()
