"""Reuse frozen joint-smoke verification plus explicit bounded-backoff checks."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
RUN='focus-joint-backoff-smoke-v1'


def verify(folder):
    path=ROOT/'scripts/verify-focus-joint-smoke.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='a685b961f293e9cc6d168ebee7f8c9e0bbe06ed488ad44e4e1ee1e05c974cee1':raise ValueError('Exact original comprehensive smoke verifier required')
    source=path.read_text(encoding='utf-8').replace('focus-joint-smoke-v1',RUN).replace('focus_joint_smoke','focus_joint_backoff_smoke')
    source=source.replace('build-focus-joint-smoke.py','build-focus-joint-backoff-smoke.py').replace('from research.focus_joint_probe import SETTINGS','from focus_joint_probe_backoff import SETTINGS')
    scope={'__file__':str(Path(__file__)),'__name__':'_frozen_backoff_verification'}
    exec(compile(source,str(path),'exec'),scope)
    receipt=scope['verify'](folder);worker=receipt['worker'];scale=worker['settings']['amp_initial_scale'];skips=0
    for row in worker['steps']:
        attempts=row['attempts'];retries=row['overflow_retries']
        if not 1<=attempts<=worker['settings']['max_amp_attempts'] or len(retries)!=attempts-1:raise ValueError('Bounded actual attempts required')
        for index,retry in enumerate(retries,1):
            if (retry['attempt']!=index or retry['optimizer_skipped'] is not True or retry['parameters_unchanged'] is not True
                or retry['scale_before']!=scale or retry['scale_after']!=scale*.5):raise ValueError('Actual overflow skip/backoff trace failed')
            scale=retry['scale_after'];skips+=1
        if row['successful_scale']!=scale:raise ValueError('Successful update used unexpected scaler state')
    receipt.update(verified_overflow_skips=skips,final_successful_scale=scale,guard_disabled=False)
    return receipt


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite actual verification')
    receipt=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}');target.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
