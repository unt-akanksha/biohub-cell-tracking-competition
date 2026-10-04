"""Recover exact terminal batch files using the previously verified harvester."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--mode', choices=('smoke', 'full'), required=True)
    args = p.parse_args()
    name = 'trajectory-event-source-v1-b' + str(args.batch)
    launch = json.loads((ROOT / 'reports/experiments' / (name + '-' + args.mode + '-launch.json')).read_text())
    assert launch['status'] == 'remote_process_started'
    base = ROOT / 'scripts/harvest-trajectory-disagreement-source-v1.py'
    assert hashlib.sha256(base.read_bytes()).hexdigest() == 'df056bca671d405b0e5c0c814632217b9254c7d53ee3b32290ab7cab444f822f'
    source = base.read_text(encoding='utf-8')
    source = source.replace('/dev/shm/biohub-disagreement-source-v1.nRsU8k/', launch['remote_root'] + '/')
    source = source.replace("result['run_id']=='trajectory-disagreement-source-v1'",
                            "result['run_id']=='trajectory-event-source-v1-batch-" + str(args.batch) + "'")
    source = source.replace('trajectory-disagreement-source-v1', name)
    source = source.replace("'complete_prelabel_predictions','failed'", "'complete_prelabel_predictions','failed','timeout'")
    sys.argv = [str(base), '--mode', args.mode]
    exec(compile(source, str(base), 'exec'), dict(__name__='__main__', __file__=str(base)))


if __name__ == '__main__':
    main()
