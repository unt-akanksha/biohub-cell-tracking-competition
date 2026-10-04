"""One-shot wait, full artifact harvest and local CPU score; never launches GPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.biohub/automation/focus-owned-flow-full-v1'
NOTEBOOK_SHA='5af97d02de71448af3d4f2367db307a77212fb59ae0d1389f0a6beffb7e3db9f'


def validate():
    folder=ROOT/'kaggle/biohub-focus-owned-flow-full-v1'
    if hashlib.sha256((folder/'biohub-focus-owned-flow-full-v1.ipynb').read_bytes()).hexdigest()!=NOTEBOOK_SHA:
        raise ValueError('Frozen full diagnostic notebook changed')
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    if (meta['id']!='indarkarhana/biohub-focus-owned-flow-full-v1' or meta['enable_gpu'] is not True
        or meta['enable_internet'] is not False or meta['enable_tpu'] is not False):
        raise ValueError('Offline frozen GPU metadata changed')
    executable=ROOT/'.biohub/cache/graph-analysis-venv/Scripts/python.exe'
    if not executable.is_file(): raise ValueError('Dedicated working CPU graph environment required')
    return executable


def run():
    python=validate()
    for name in ('OMP_NUM_THREADS','POLARS_MAX_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='2'
    spec=importlib.util.spec_from_file_location('focus_harvest_helpers',ROOT/'scripts/run-owned-detector-evaluation-queue.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper); helper.STATE=STATE
    ref='indarkarhana/biohub-focus-owned-flow-full-v1'
    helper.wait_complete(ref,time.monotonic()+2*3600)
    helper.harvest(ref,'focus-owned-flow-full-v1',r'.*')
    validate()
    helper.log('local_cpu_score_started',ref=ref,notebook_sha256=NOTEBOOK_SHA)
    helper.command([python,ROOT/'scripts/score-focus-owned-flow-full.py'],timeout=1800)
    result=json.loads((ROOT/'reports/experiments/focus-owned-flow-full-v1-result.json').read_text())
    helper.log('full_cpu_score_completed',comparison=result['comparison'],summaries=result['summaries'],
        submission_performed=False,new_target_movies_opened=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); validate()
    if not args.execute:
        print('Frozen full diagnostic and CPU environment verified; no external action'); raise SystemExit(0)
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
            submission_performed=False,new_target_movies_opened=0)),encoding='utf-8')
