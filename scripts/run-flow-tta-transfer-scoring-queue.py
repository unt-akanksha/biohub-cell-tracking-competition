"""One-shot CPU-only follow-up to the already launched frozen transfer job."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.biohub/automation/flow-tta-transfer-v1'
PINS={'flow-tta-transfer-v1':'76e3075c9b6868cc47408f8d6be4d7a4551000d2eeb448ac94776d0890668b80',
      'flow-tta-transfer-scoring-v1':'fa71524b0bc23ed23f5cf4ac23c91a780ccde0d233ba0d917b1bb1b9ee865040'}


def validate_staging():
    for run,sha in PINS.items():
        folder=ROOT/'kaggle'/('biohub-'+run)
        if hashlib.sha256((folder/('biohub-'+run+'.ipynb')).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen transfer notebook changed')
        meta=json.loads((folder/'kernel-metadata.json').read_text())
        if (meta['id']!='indarkarhana/biohub-'+run or meta['enable_internet'] is not False
            or meta['enable_tpu'] is not False or meta['enable_gpu'] is not ('scoring' not in run)
            or max(len(meta['title']),len(meta['id'].split('/')[1]))>50):
            raise ValueError('Exact offline short-name transfer metadata required')


def run():
    validate_staging()
    spec=importlib.util.spec_from_file_location('transfer_queue_helpers',ROOT/'scripts/run-owned-detector-evaluation-queue.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper); helper.STATE=STATE
    gpu='indarkarhana/biohub-flow-tta-transfer-v1'; cpu='indarkarhana/biohub-flow-tta-transfer-scoring-v1'
    deadline=time.monotonic()+2*3600
    helper.wait_complete(gpu,deadline)
    helper.harvest(gpu,'flow-tta-transfer-v1',r'(selection_manifest.json|launcher_terminal.json|source_hashes.json|runtime_hashes.json)$')
    validate_staging()
    if helper.INSPECT(cpu)['present']: raise ValueError('Scorer exists; no duplicate launch')
    helper.log('cpu_push_intent',ref=cpu,notebook_sha256=PINS['flow-tta-transfer-scoring-v1'])
    receipt=helper.command(['kaggle','kernels','push','-p',ROOT/'kaggle/biohub-flow-tta-transfer-scoring-v1','--timeout','3600'])
    helper.verify_push(receipt,cpu); helper.log('cpu_push_accepted',ref=cpu,version=1,gpu=False)
    helper.wait_complete(cpu,deadline)
    helper.harvest(cpu,'flow-tta-transfer-scoring-v1',r'(selection_score.json|launcher_terminal.json)$')
    output=helper.command([sys.executable,ROOT/'scripts/summarize-flow-tta-transfer.py'])
    helper.log('transfer_comparison_completed',comparison=json.loads(output),submission_performed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); validate_staging()
    if not args.execute:
        print('Frozen transfer staging verified; no external action'); raise SystemExit(0)
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
