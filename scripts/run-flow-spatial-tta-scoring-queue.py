"""Bounded one-shot CPU scoring after the already launched motion GPU job."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.biohub/automation/flow-spatial-tta-selection-v1'
PINS={'flow-spatial-tta-selection-v1':'ea5d8fc88c031415159211bb212ee826d2ef4ad14995f782757219d22755498f',
      'flow-spatial-tta-scoring-v1':'5ad4766e4b394364ec0d3963c9227b16699c92c5879671020a6b813371509daf'}


def validate_staging():
    for run,sha in PINS.items():
        folder=ROOT/'kaggle'/('biohub-'+run)
        if hashlib.sha256((folder/('biohub-'+run+'.ipynb')).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen motion notebook changed')
        meta=json.loads((folder/'kernel-metadata.json').read_text())
        if (meta['id']!='indarkarhana/biohub-'+run or meta['enable_internet'] is not False
            or meta['enable_tpu'] is not False or meta['enable_gpu'] is not ('selection' in run)
            or len(meta['title'])>50 or len(meta['id'].split('/')[1])>50):
            raise ValueError('Short-name offline resource metadata required')


def run():
    validate_staging()
    spec=importlib.util.spec_from_file_location('queue_helpers',ROOT/'scripts/run-owned-detector-evaluation-queue.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper); helper.STATE=STATE
    selection='indarkarhana/biohub-flow-spatial-tta-selection-v1'
    scoring='indarkarhana/biohub-flow-spatial-tta-scoring-v1'
    deadline=time.monotonic()+2*3600
    helper.wait_complete(selection,deadline)
    helper.harvest(selection,'flow-spatial-tta-selection-v1',r'(selection_manifest.json|launcher_terminal.json|source_hashes.json|runtime_hashes.json)$')
    validate_staging()
    if helper.INSPECT(scoring)['present']: raise ValueError('CPU scorer already exists; no duplicate launch')
    helper.log('cpu_push_intent',ref=scoring,notebook_sha256=PINS['flow-spatial-tta-scoring-v1'])
    output=helper.command(['kaggle','kernels','push','-p',ROOT/'kaggle/biohub-flow-spatial-tta-scoring-v1','--timeout','3600'])
    helper.verify_push(output,scoring)
    helper.log('cpu_push_accepted',ref=scoring,version=1,gpu=False)
    helper.wait_complete(scoring,deadline)
    helper.harvest(scoring,'flow-spatial-tta-scoring-v1',r'(selection_score.json|launcher_terminal.json)$')
    output=helper.command([sys.executable,ROOT/'scripts/summarize-flow-spatial-tta-selection.py'])
    helper.log('source_comparison_completed',comparison=json.loads(output),submission_performed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); validate_staging()
    if not args.execute:
        print('Frozen short-name staging verified; no external action'); raise SystemExit(0)
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
