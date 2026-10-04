"""Sequential source collection controller, no model fitting or submission changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('C:/Users/IndarKumar/AppData/Local/Programs/Python/Python312/python.exe')
ANALYSIS = ROOT / '.biohub/cache/graph-analysis-venv/Scripts/python.exe'
SSH = ['-i','C:/Users/IndarKumar/.ssh/rsna_ec2','-o','BatchMode=yes','-o','ConnectTimeout=15',
       '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=13.220.240.128']
HOST = 'ubuntu@3.226.249.134'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_status(status):
    if status == 'running':
        return False
    if status in ('complete_prelabel_predictions', 'failed', 'timeout'):
        return True
    raise ValueError('Unexpected collection state: ' + str(status))


def remote_state(launch):
    root, pid = launch['remote_root'], launch['remote']['pid']
    code = '''import json
from pathlib import Path
root=Path(''' + repr(root) + ''');pid=''' + repr(pid) + '''
terminal=root/'full/result.json'
if terminal.exists():
 try: result=json.loads(terminal.read_text())
 except json.JSONDecodeError: result=None
else: result=None
if result is None:
 try: command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
 except FileNotFoundError: command=b''
 assert str(root).encode() in command and b'run-trajectory-division-full-movie-v1.py' in command, 'Live handle missing without terminal; inspect, do not restart'
 result=dict(status='running',pid=pid,movies={})
 progress=root/'full/prelabel-progress.json'
 if progress.exists():
  try: result['movies']=json.loads(progress.read_text()).get('movies',{})
  except json.JSONDecodeError: pass
print(json.dumps(dict(status=result['status'],movies=list(result.get('movies',{})),
 elapsed_seconds=result.get('elapsed_seconds'),contract_sha256=result.get('contract_sha256'))))
'''
    response = subprocess.run(['ssh',*SSH,HOST,'/home/ubuntu/venv/bin/python -'],
                              input=code,text=True,capture_output=True,timeout=30,check=True)
    return json.loads(response.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--active-batch',type=int,required=True)
    p.add_argument('--through-batch',type=int,required=True)
    args = p.parse_args()
    assert 0 <= args.active_batch <= args.through_batch <= 7
    receipt = ROOT / 'reports/experiments/trajectory-event-source-v1-controller.json'
    assert not receipt.exists(), 'Controller already attempted; inspect active job before any resumption'
    plan = read(ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json')
    first = read(ROOT / 'reports/experiments/trajectory-event-source-v1-b0-capacity.json')
    assert first['source_only'] and first['pooled']['all_three_cells_matched'] > 0
    assert read(ROOT / 'reports/experiments/trajectory-event-source-v1-b0-full-harvest.json')['status'] == 'verified_backup'
    assert plan['frozen_submitted_evaluation_is_not_a_parallel_experiment']
    state = dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                 active_batch=args.active_batch,through_batch=args.through_batch,completed=[],
                 maximum_new_gpu_hours=(args.through_batch-args.active_batch)*.75,
                 source_plan_sha256=sha(ROOT / 'reports/experiments/trajectory-event-source-v1-plan.json'),
                 model_fitting_or_submission_authorized=False,foreign_jobs_must_not_be_interrupted=True)
    def checkpoint(phase, **values):
        state.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**values)
        receipt.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(event=phase,active_batch=state['active_batch'],**values)),flush=True)
    def run(command, timeout=300):
        subprocess.run(list(map(str,command)),check=True,timeout=timeout,cwd=ROOT)
    def script(filename, batch, *extra, analysis=False, timeout=300):
        run([ANALYSIS if analysis else BASE,ROOT/'scripts'/filename,'--batch',batch,*extra],timeout=timeout)
    checkpoint('controller_started')
    try:
        for batch in range(args.active_batch,args.through_batch+1):
            name = 'trajectory-event-source-v1-b'+str(batch)
            state['active_batch'] = batch
            launch_path = ROOT/'reports/experiments'/(name+'-full-launch.json')
            if batch != args.active_batch:
                assert not launch_path.exists(), 'Never duplicate a recorded launch'
                checkpoint('preparing_next_batch')
                row = next(b for b in plan['batches'] if b['index']==batch)
                scope = ROOT/'.biohub/cache/trajectory-event-source-v1-plan'/('batch-'+str(batch))
                assert sha(scope/'MOVIES.json')==row['plan_sha256']
                script('prepare-trajectory-event-archive-v1.py',batch)
                private = read(scope/'PRIVATE_ARCHIVE_PLAN.json')
                assert private['total_image_bytes']<=row['maximum_image_cache_bytes']
                made = subprocess.run(['ssh',*SSH,HOST,'mktemp -d /dev/shm/biohub-event-source-v1-b'+str(batch)+'.XXXXXX'],
                                      capture_output=True,text=True,check=True,timeout=30)
                remote = made.stdout.strip()
                assert re.fullmatch(r'/dev/shm/biohub-event-source-v1-b'+str(batch)+r'\.[A-Za-z0-9]{6}',remote)
                checkpoint('remote_batch_created',remote_root=remote)
                run(['scp',*SSH,ROOT/'scripts/download-trajectory-event-source-v1.py',
                     ROOT/'scripts/download-trajectory-division-archive-v1.py',ROOT/'research/kaggle_archive_ranges.py',
                     scope/'MOVIES.json',scope/'PRIVATE_ARCHIVE_PLAN.json',HOST+':'+remote+'/'])
                command = '/home/ubuntu/venv/bin/python '+remote+'/download-trajectory-event-source-v1.py'
                command += ' --plan '+remote+'/PRIVATE_ARCHIVE_PLAN.json --plan-sha256 '+sha(scope/'PRIVATE_ARCHIVE_PLAN.json')
                command += ' --helper '+remote+'/kaggle_archive_ranges.py --helper-sha256 '+sha(ROOT/'research/kaggle_archive_ranges.py')
                command += ' --output '+remote+'/images'
                checkpoint('downloading_verified_images')
                run(['ssh',*SSH,HOST,command],timeout=1260)
                run(['scp',*SSH,HOST+':'+remote+'/images/IMAGE_MANIFEST.json',scope/'IMAGE_MANIFEST.json'])
                script('package-trajectory-event-source-v1.py',batch)
                run(['scp',*SSH,ROOT/'.biohub/cache'/(name+'-bundle.tar'),HOST+':'+remote+'/'])
                checkpoint('launching_smoke')
                script('launch-trajectory-event-source-v1.py',batch,'--mode','smoke','--remote-root',remote)
                script('advance-trajectory-event-source-v1.py',batch,timeout=960)
            launch = read(launch_path)
            assert launch['status']=='remote_process_started' and launch['mode']=='full'
            checkpoint('waiting_for_full_batch',remote_root=launch['remote_root'],pid=launch['remote']['pid'])
            last_movies = None
            while True:
                observed = remote_state(launch)
                if terminal_status(observed['status']):
                    break
                if observed['movies'] != last_movies:
                    last_movies = observed['movies']
                    checkpoint('full_batch_progress',finished_movies=last_movies)
                time.sleep(30)
            checkpoint('harvesting_terminal_batch',terminal_status=observed['status'])
            script('harvest-trajectory-event-source-v1.py',batch,'--mode','full',timeout=240)
            if observed['status']!='complete_prelabel_predictions':
                raise RuntimeError('Batch failed or timed out; recovered artifacts, no restart or next batch')
            assert observed['contract_sha256']==launch['contract_sha256']
            checkpoint('freezing_source_features')
            script('prepare-trajectory-event-features-v1.py',batch,analysis=True,timeout=900)
            checkpoint('auditing_source_capacity')
            script('audit-trajectory-event-batch-capacity-v1.py',batch,analysis=True,timeout=900)
            # Counts are diagnostic, never used to discard difficult movies.
            checkpoint('retiring_recoverable_images')
            script('retire-trajectory-event-images-v1.py',batch)
            state['completed'].append(batch)
            checkpoint('batch_complete')
        state['status']='complete'
        checkpoint('all_requested_source_batches_complete')
    except BaseException as error:
        state.update(status='stopped_requires_inspection',error_type=type(error).__name__)
        checkpoint('controller_stopped_no_automatic_restart')
        raise


if __name__=='__main__':
    main()
