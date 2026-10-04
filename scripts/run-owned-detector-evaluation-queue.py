"""Bounded sequential evaluation queue; never submits or touches AWS/other jobs.

Run once after tests. Any failure stops the queue, preserving remote artifacts.
No retries of ambiguous pushes, no relaunches, no target-embryo access.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'.biohub/automation/owned-detector-evaluation-v1'
FIT = 'indarkarhana/biohub-owned-detector-fit-pair-v1'
INSPECT = runpy.run_path(str(ROOT/'scripts/get-kaggle-kernel-state.py'))['inspect_owned_kernel_state']


def gpu_remaining(output):
    rows = re.findall(r'^GPU\s+[0-9.]+h\s+([0-9.]+)h\s+[0-9.]+h\s+',output,re.M)
    if len(rows)!=1: raise ValueError('Cannot establish fresh GPU remaining quota')
    remaining = float(rows[0])
    if not 9 <= remaining <= 1000:
        raise ValueError('One-hour job would violate the eight-hour GPU reserve')
    return remaining


def status_value(output, ref):
    match = re.fullmatch(re.escape(ref)+r' has status "KernelWorkerStatus\.([A-Z_]+)"',output.strip())
    if not match: raise ValueError('Unrecognized authoritative kernel status')
    status = match[1]
    if status not in ('RUNNING','QUEUED','COMPLETE'):
        raise RuntimeError(f'{ref}: terminal or unsupported status {status}; no automatic relaunch')
    return status


def verify_push(output, ref):
    versions = re.findall(r'Kernel version\s+(\d+)\s+successfully pushed',output,re.I)
    if versions!=['1'] or f'https://www.kaggle.com/code/{ref}' not in output:
        raise RuntimeError('Push receipt ambiguous or wrong version/slug; stop without retry')


def log(event, **values):
    record = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=event,**values)
    print(json.dumps(record),flush=True)
    with (STATE/'events.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(record)+'\n')


def command(args, timeout=300):
    env = dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    proc = subprocess.run([str(x) for x in args],cwd=ROOT,env=env,capture_output=True,
                          encoding='utf-8',errors='replace',timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'Command failed ({proc.returncode}): {args[0:3]}\n{proc.stdout[-2000:]}\n{proc.stderr[-1000:]}')
    return proc.stdout


def wait_complete(ref, deadline):
    previous = None
    while time.monotonic()<deadline:
        status = status_value(command(['kaggle','kernels','status',ref]),ref)
        if status!=previous: log('kernel_status',ref=ref,status=status)
        previous = status
        if status=='COMPLETE':
            state = INSPECT(ref)
            if not state['present'] or state['current_version_number']!=1:
                raise ValueError('Remote artifact version changed')
            return
        time.sleep(60)
    raise TimeoutError('Bounded evaluation queue expired; existing jobs left untouched')


def harvest(ref, cache_name, pattern):
    target = ROOT/'.biohub/cache/kernel-outputs'/cache_name
    command(['kaggle','kernels','output',ref+'/1','-p',target,'--file-pattern',pattern,'-q'])
    log('harvested_receipts',ref=ref,cache=str(target))


def launch(arm, kind, gpu):
    name = f'biohub-owned-detector-{arm}-{kind}-v1'
    ref, folder = 'indarkarhana/'+name, ROOT/'kaggle'/name
    state = INSPECT(ref)
    if state['present']: raise ValueError(f'{ref} already exists; do not rebuild or launch another version')
    builder = 'selection' if kind=='selection' else 'scoring'
    command([sys.executable,ROOT/f'scripts/build-owned-detector-{builder}.py','--arm',arm])
    nbpath = folder/(name+'.ipynb')
    nb = json.loads(nbpath.read_text())
    meta = json.loads((folder/'kernel-metadata.json').read_text())
    if (meta['id']!=ref or meta['enable_gpu'] is not gpu or meta['enable_internet'] is not False
        or meta['enable_tpu'] is not False):
        raise ValueError('Offline resource metadata changed')
    args = ['kaggle','kernels','push','-p',folder,'--timeout','3600']
    if gpu:
        if nb['metadata']['codex']['declared_budget_seconds']!=3600:
            raise ValueError('Missing one-hour inference budget')
        remaining = gpu_remaining(command(['kaggle','quota']))
        log('fresh_quota_authorization',ref=ref,remaining_hours=remaining,worst_case_hours=1,reserve_hours=8)
        args += ['--accelerator','NvidiaTeslaT4']
    # Record intent BEFORE the external write. Never retry a timeout or ambiguous response.
    log('push_intent',ref=ref,notebook_sha256=hashlib.sha256(nbpath.read_bytes()).hexdigest())
    receipt = command(args)
    verify_push(receipt,ref)
    log('push_accepted',ref=ref,version=1,gpu=gpu)
    return ref


def run():
    deadline = time.monotonic()+3*3600
    wait_complete(FIT,deadline)
    harvest(FIT,'owned-detector-fit-pair-v1',r'(outputs/pair_result.json|outputs/(sparse|pu)/result.json|launcher_terminal.json|source_hashes.json|runtime_hashes.json)$')
    command([sys.executable,ROOT/'scripts/summarize-owned-detector-fit-pair.py'])
    log('paired_training_verified')
    cpu_jobs = []
    for arm in ('sparse','pu'):
        selection = launch(arm,'selection',True)
        wait_complete(selection,deadline)
        harvest(selection,f'owned-detector-{arm}-selection-v1',r'(selection_manifest.json|launcher_terminal.json|source_hashes.json|runtime_hashes.json)$')
        cpu_jobs.append((arm,launch(arm,'scoring',False)))
    for arm,ref in cpu_jobs:
        wait_complete(ref,deadline)
        harvest(ref,f'owned-detector-{arm}-scoring-v1',r'(selection_score.json|launcher_terminal.json)$')
    output = command([sys.executable,ROOT/'scripts/summarize-owned-detector-selection.py'])
    log('source_comparison_completed',comparison=json.loads(output),submission_performed=False,target_audit_opened=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Validation-only: run pytest tests/test_owned_detector_evaluation_queue.py before --execute')
        raise SystemExit(0)
    STATE.mkdir(parents=True,exist_ok=True)
    # Permanent one-shot receipt prevents duplicate controllers, including after failure.
    with (STATE/'started.json').open('x',encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),handle)
    status = 'failed'
    try:
        log('started',pid=os.getpid(),maximum_seconds=10800)
        run()
        status = 'completed'
    except Exception as exc:
        log('failed',error=str(exc),submission_performed=False)
        raise
    finally:
        (STATE/'terminal.json').write_text(json.dumps(dict(status=status,submission_performed=False,target_audit_opened=False)),encoding='utf-8')
