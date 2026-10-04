"""Verify actual runtime/checkpoints and recompute the frozen diagnostic gate."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_adaptation_training import SETTINGS,diagnostic_gate
RUN='focus-adaptation-head-v1'
NOTEBOOK_SHA='e2df68e7e4355840197d42f0c18c9a609c31f1786a32a9b5b74f861dad8a9288'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    path=ROOT/f'kaggle/biohub-{RUN}/biohub-{RUN}.ipynb'
    if sha(path)!=NOTEBOOK_SHA:raise ValueError('Exact launched head-adaptation notebook required')
    nb=json.loads(path.read_text())
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);spec=json.loads(runtime['training_spec.json'])
    if spec!=runpy.run_path(str(ROOT/'scripts/build-focus-adaptation-head.py'))['make_spec']():
        raise ValueError('Training contract differs from verified feature evidence')
    work=folder/'focus_adaptation_head';output=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual training runtime differs from immutable notebook')
    launcher=json.loads((work/'launcher_terminal.json').read_text());result=json.loads((output/'result.json').read_text())
    if (launcher['status']!='completed' or launcher['run_id']!=RUN or launcher['declared_budget_seconds']!=3600
        or not 0<launcher['elapsed_seconds']<=3600 or launcher['submission_performed'] is not False):
        raise ValueError('Completed bounded launcher required')
    if (result['status']!='completed_focus_head_adaptation' or result['settings']!=SETTINGS
        or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['frozen_before']!=result['frozen_after'] or result['initial_model_sha256']!=spec['model_tensor_sha256']
        or result['initial_model_sha256']==result['final_model_sha256']
        or result['all_diagnostic_samples_excluded_from_optimizer'] is not True or result['new_target_movies_opened']!=0
        or any(result[k] is not False for k in ('raw_detections_changed','source_selection_opened','authorized_for_submission'))):
        raise ValueError('Fixed head-only training identity or guards failed')
    for key,step in [('smoke',4),('checkpoint',800)]:
        record=result[key]
        if record['file']!=f'step-{step:04d}.pt' or record['exact_real_logit_reload'] is not True or sha(output/record['file'])!=record['sha256']:
            raise ValueError('Actual restricted-reload checkpoint proof differs')
    diagnostic=[r for r in spec['feature_records'] if r['role']=='diagnostic']
    parent=sum(r['counts']['known_parent'] for r in diagnostic);absent=sum(r['counts']['known_absent'] for r in diagnostic)
    for name in ('physical_diagnostic','initial_diagnostic','final_diagnostic'):
        row=result[name]
        if row['known_parent']!=parent or row['known_absent']!=absent or not 0<=row['correct_parent']<=parent or not 0<=row['correct_absent']<=absent:
            raise ValueError('Whole diagnostic supervision changed between arms')
        if abs(row['nll']-row['loss_sum']/(parent+absent))>1e-12:raise ValueError('Micro-averaged diagnostic loss mismatch')
    gate=diagnostic_gate(result['initial_diagnostic'],result['physical_diagnostic'],result['final_diagnostic'])
    if gate!=result['diagnostic_gate']:raise ValueError('Actual result differs from unchanged feasibility gate')
    return dict(status='verified_focus_head_adaptation',run_id=RUN,notebook_sha256=NOTEBOOK_SHA,
        worker_result_sha256=sha(output/'result.json'),launcher=launcher,worker=result,diagnostic_gate=gate,
        eligible_for_source_tracking_evaluation=gate['passed'],authorized_for_submission=False,
        caveat='Training-domain diagnostic only; original encoder/head training included diagnostic movies. No independent or public leaderboard claim.')


if __name__=='__main__':
    path=ROOT/f'reports/experiments/{RUN}-result.json'
    if path.exists():raise ValueError('Never overwrite actual verified experiment')
    result=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}')
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
