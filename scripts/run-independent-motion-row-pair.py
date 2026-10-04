"""Sequential paired 100-step fits with identical augmented input fingerprints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo','runtime','checkpoint','manifest','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--steps',type=int,choices=(100,1000),default=100)
    parser.add_argument('--training-scope',choices=('pilot','full'),default='pilot')
    parser.add_argument('--motion-residual',action='store_true')
    parser.add_argument('--row-negatives',action='store_true')
    args = parser.parse_args()
    results = {}
    for arm in ('control','row'):
        command = [sys.executable,'-u',str(args.runtime/'run_association.py'),
            '--sha256',args.sha256,'--steps',str(args.steps),'--motion-residual',
            '--training-scope',args.training_scope]
        for name in ('repo','runtime','checkpoint','manifest','data'):
            command.extend(['--'+name,str(getattr(args,name))])
        command.extend(['--output',str(args.output/arm)])
        if arm == 'row':
            command.append('--row-negatives')
        subprocess.run(command,check=True,timeout=1500)
        results[arm] = json.loads((args.output/arm/'result.json').read_text())
    if results['control']['sample_hashes'] != results['row']['sample_hashes']:
        raise ValueError('Paired training inputs differ')
    if len(results['control']['sample_hashes']) != 2*args.steps:
        raise ValueError('Unexpected paired sample coverage')
    if results['control']['frozen_detector_sha256'] != results['row']['frozen_detector_sha256']:
        raise ValueError('Paired detector mismatch')
    (args.output/'paired_result.json').write_text(json.dumps(dict(
        status='completed',inputs_identical=True,sample_count=2*args.steps,results=results,
        authorized_for_submission=False),indent=2))
    print(json.dumps(dict(event='paired_gate',inputs_identical=True,sample_count=2*args.steps)),flush=True)


if __name__ == '__main__':
    main()
