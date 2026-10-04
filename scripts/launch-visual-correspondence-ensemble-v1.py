"""Launch only the hash-bound Biohub job in the existing owned cloud directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
parser.add_argument('--revision', type=int, choices=(1, 2), default=1)
args = parser.parse_args()
root = Path('/tmp/biohub-image-context-v2.ScdSdY').resolve(strict=True)
suffix = 'v1' if args.revision == 1 else 'v1-r2'
bundle = (root/f'visual-correspondence-ensemble-{suffix}-bundle').resolve(strict=True)
expected = {1: '096abae5977aa95d68091cea49bafcdeed98bb879578b056e421b0472c291d4a',
            2: 'a3a59450791c4d6aef7e75f8d7e2a40c5a41f9dc1a4a14d0440f0a1f0e1712d0'}[args.revision]
if root not in bundle.parents or hashlib.sha256((bundle/'BUNDLE.json').read_bytes()).hexdigest() != expected:
    raise ValueError('Exact owned bundle required')
if args.revision == 2 and not (bundle/'data').exists():
    import os
    previous = (root/'visual-correspondence-ensemble-v1-bundle').resolve(strict=True)
    contract = json.loads((bundle/'BUNDLE.json').read_text())
    for record in contract['records']:
        source = (previous/record['path']).resolve(strict=True)
        target = bundle/record['path']
        if previous not in source.parents or bundle not in target.resolve().parents:
            raise ValueError('Shared data escaped exact bundle roots')
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Shared data hash mismatch')
        target.parent.mkdir(exist_ok=True)
        os.link(source, target)
output = root/f'visual-correspondence-{args.mode}-{suffix}'
receipt = root/f'visual-correspondence-{args.mode}-{suffix}-launch.json'
if output.exists() or receipt.exists():
    raise ValueError('Do not restart/overwrite an existing job')
command = [sys.executable, '-u', str(bundle/'run.py'), '--bundle', str(bundle),
           '--bundle-sha256', expected, '--mode', args.mode, '--output', str(output)]
if args.mode == 'train':
    proof = root/f'visual-correspondence-smoke-{suffix}/result.json'
    result = json.loads(proof.read_text())
    if result['status'] != 'functionality_passed' or result['bundle_sha256'] != expected:
        raise ValueError('Passing matching real-data smoke required')
    command += ['--smoke-proof', str(proof)]
log_path = root/f'visual-correspondence-{args.mode}-{suffix}.log'
with log_path.open('xb') as stream:
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
result = dict(pid=child.pid, mode=args.mode, started_epoch=time.time(), command=command,
              bundle_sha256=expected, log=str(log_path), output=str(output))
receipt.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
