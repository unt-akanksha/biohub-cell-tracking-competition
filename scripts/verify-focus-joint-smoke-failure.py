"""Preserve actual failed runtime and uniquely reported smoke progress."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUN='focus-joint-smoke-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    notebook=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    if sha(notebook)!='87f68814914a35dfa71b2e4015f8ddf1e392b7e68d8be42654dee2054c0f1f30':raise ValueError('Exact failed notebook required')
    nb=json.loads(notebook.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);folder=ROOT/f'.biohub/cache/kernel-outputs/{RUN}';work=folder/'focus_joint_smoke'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual failed runtime changed')
    terminal_path=work/'launcher_terminal.json';terminal=json.loads(terminal_path.read_text())
    if terminal['status']!='error' or terminal['run_id']!=RUN or not 0<terminal['elapsed_seconds']<=3600 or terminal['submission_performed'] is not False:raise ValueError('Actual bounded failure terminal required')
    log_path=folder/f'biohub-{RUN}.log';stream=''.join(r['data'] for r in json.loads(log_path.read_text(encoding='utf-8')))
    records={}
    for line in stream.splitlines():
        try:row=json.loads(line)
        except (ValueError,TypeError):continue
        if isinstance(row,dict) and row.get('stage') in ('all_four_original_image_replays_passed','joint_smoke_step'):
            records[json.dumps(row,sort_keys=True)]=row
    replays=[r for r in records.values() if r['stage']=='all_four_original_image_replays_passed']
    steps=[r for r in records.values() if r['stage']=='joint_smoke_step']
    if len(replays)!=1 or len(replays[0]['pairs'])!=4 or [r['step'] for r in steps]!=[1] or 'The total norm of order 2.0 for gradients' not in stream or 'is non-finite' not in stream:raise ValueError('Reported failure boundary differs')
    return dict(status='verified_joint_smoke_nonfinite_gradient_failure',run_id=RUN,notebook_sha256=sha(notebook),terminal=terminal,
        terminal_sha256=sha(terminal_path),log_sha256=sha(log_path),exact_original_pair_replays=replays[0]['pairs'],completed_updates=steps,
        failure='Nonfinite unscaled gradient norm before the second optimizer update',
        scale_overflow_hypothesis_unproven=True,eligible_for_larger_training=False,authorized_for_submission=False)


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-failure.json'
    if target.exists():raise ValueError('Never overwrite actual failure record')
    result=verify();target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
