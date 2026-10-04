"""Verify one already-launched smoke, back it up, then launch its full batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.trajectory_runtime_v1 import sha, validate_graph

SSH = ['-i', 'C:/Users/IndarKumar/.ssh/rsna_ec2', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=13.220.240.128']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    args = p.parse_args()
    name = 'trajectory-event-source-v1-b' + str(args.batch)
    full_receipt = ROOT / 'reports/experiments' / (name + '-full-launch.json')
    assert not full_receipt.exists(), 'Full launch already attempted; inspect actual state'
    launch = json.loads((ROOT / 'reports/experiments' / (name + '-smoke-launch.json')).read_text())
    assert launch['status'] == 'remote_process_started'
    root, pid = launch['remote_root'], launch['remote']['pid']
    code = '''import json
from pathlib import Path
root=Path(''' + repr(root) + ''');pid=''' + repr(pid) + '''
path=root/'smoke/result.json'
if path.exists():
 try: result=json.loads(path.read_text())
 except json.JSONDecodeError: result=dict(status='writing_terminal')
else:
 proc=Path('/proc')/str(pid)
 try: command=(proc/'cmdline').read_bytes()
 except FileNotFoundError: command=b''
 assert str(root).encode() in command, 'Smoke handle missing without terminal; inspect logs'
 result=dict(status='running',pid=pid)
print(json.dumps(result))
'''
    deadline = time.monotonic() + 750
    while True:
        response = subprocess.run(['ssh', *SSH, 'ubuntu@3.226.249.134', '/home/ubuntu/venv/bin/python -'],
                                  input=code, text=True, capture_output=True, timeout=30)
        if response.returncode:
            raise RuntimeError(response.stderr)
        result = json.loads(response.stdout)
        if result['status'] not in ('running', 'writing_terminal'):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Observation deadline reached; inspect same smoke, never restart automatically')
        time.sleep(5)
    subprocess.run([sys.executable, str(ROOT / 'scripts/harvest-trajectory-event-source-v1.py'),
                    '--batch', str(args.batch), '--mode', 'smoke'], check=True, timeout=180)
    assert result['status'] == 'functionality_passed' and result['inputs_unchanged']
    assert result['contract_sha256'] == launch['contract_sha256'] and not result['ground_truth_opened']
    local = ROOT / '.biohub/cache' / (name + '-smoke-output')
    for movie, arms in result['movies'].items():
        assert set(arms) == {'original'} and arms['original']['frames'] == 8
        for filename, key in (('prediction.json', 'prediction_sha256'), ('repaired-prediction.json', 'repaired_sha256')):
            path = local / (movie + '-original') / filename
            assert sha(path) == arms['original'][key]
            validate_graph(json.loads(path.read_text()), 8)
    print(json.dumps(dict(event='smoke_backed_up_and_graphs_verified', batch=args.batch,
                          smoke_seconds=result['elapsed_seconds'])), flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/launch-trajectory-event-source-v1.py'),
                    '--batch', str(args.batch), '--mode', 'full', '--remote-root', root], check=True, timeout=90)


if __name__ == '__main__':
    main()
