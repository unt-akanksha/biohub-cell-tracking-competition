"""One-shot CPU follow-through for the already launched ensemble inference.

Never launches GPU, submits, retries pushes, rebuilds notebooks or touches AWS.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN='owned-detector-ensemble-selection-v1'
STATE=ROOT/'.biohub/automation/owned-detector-ensemble-selection-v1'
PINS={RUN:'5dde4d126d150fb9c861df451e2b813f361b052c6bcaed93948df3aa4ee03bc1',
      RUN+'-scoring':'d7d6ba1823f723f941dcc7ab076a0e92d0e14dd449c27377064b612577bd5f99'}


def validate_staging():
    for run,sha in PINS.items():
        folder=ROOT/'kaggle'/('biohub-'+run)
        if hashlib.sha256((folder/('biohub-'+run+'.ipynb')).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen ensemble launch changed')
        meta=json.loads((folder/'kernel-metadata.json').read_text())
        if (meta['id']!='indarkarhana/biohub-'+run or meta['enable_internet'] is not False
            or meta['enable_tpu'] is not False or meta['enable_gpu'] is not (run==RUN)):
            raise ValueError('Resource/identity metadata changed')


def run():
    validate_staging()
    spec=importlib.util.spec_from_file_location('queue_helpers',ROOT/'scripts/run-owned-detector-evaluation-queue.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.STATE=STATE
    selection='indarkarhana/biohub-'+RUN
    scoring=selection+'-scoring'
    deadline=time.monotonic()+2*3600
    helper.wait_complete(selection,deadline)
    helper.harvest(selection,RUN,r'(selection_manifest.json|launcher_terminal.json|source_hashes.json|runtime_hashes.json)$')
    validate_staging()
    if helper.INSPECT(scoring)['present']:
        raise ValueError('CPU scorer already exists; no automatic duplicate launch')
    helper.log('cpu_push_intent',ref=scoring,notebook_sha256=PINS[RUN+'-scoring'])
    output=helper.command(['kaggle','kernels','push','-p',ROOT/'kaggle'/('biohub-'+RUN+'-scoring'),'--timeout','3600'])
    helper.verify_push(output,scoring)
    helper.log('cpu_push_accepted',ref=scoring,version=1,gpu=False)
    helper.wait_complete(scoring,deadline)
    helper.harvest(scoring,RUN+'-scoring',r'(selection_score.json|launcher_terminal.json)$')
    output=helper.command([sys.executable,ROOT/'scripts/summarize-owned-detector-ensemble-selection.py'])
    helper.log('source_comparison_completed',comparison=json.loads(output),submission_performed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); validate_staging()
    if not args.execute:
        print('Frozen staging verified; no external action'); raise SystemExit(0)
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'started.json').open('x',encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),handle)
    status='failed'; error=None
    try:
        run(); status='completed'
    except Exception as exc:
        error=str(exc); raise
    finally:
        (STATE/'terminal.json').write_text(json.dumps(dict(status=status,error=error,
            submission_performed=False,target_audit_opened=False)),encoding='utf-8')
