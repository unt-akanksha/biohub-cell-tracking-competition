"""Verify saved runtime/checkpoints and recompute unchanged diagnostic gates."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_adaptation_training import SETTINGS,diagnostic_gate
RUN='focus-null-balanced-head-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    staged=ROOT/f'kaggle/biohub-{RUN}';identity=json.loads((staged/'staged_identity.json').read_text())
    notebook=staged/f'biohub-{RUN}.ipynb';builder=ROOT/'scripts/build-focus-null-balanced-head.py'
    if (sha(notebook)!=identity['notebook_sha256'] or sha(staged/'kernel-metadata.json')!=identity['metadata_sha256']
        or sha(builder)!=identity['builder_sha256']):raise ValueError('Frozen staged identity changed')
    nb=json.loads(notebook.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);spec=json.loads(runtime['training_spec.json'])
    if spec!=runpy.run_path(str(builder))['make_spec']():raise ValueError('Upstream verified evidence changed')
    work=folder/'focus_null_balanced_head';output=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual runtime changed')
    launcher=json.loads((work/'launcher_terminal.json').read_text());result=json.loads((output/'result.json').read_text())
    if (launcher['status']!='completed' or launcher['run_id']!=RUN or launcher['declared_budget_seconds']!=3600
        or not 0<launcher['elapsed_seconds']<=3600 or launcher['submission_performed'] is not False):raise ValueError('Bounded completed launcher required')
    if (result['status']!='completed_focus_null_balanced_head' or result['settings']!=SETTINGS
        or result['null_weight']!=spec['null_weight'] or result['supervision_counts']!=spec['supervision_counts']
        or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['frozen_before']!=result['frozen_after'] or result['initial_model_sha256']!=spec['model_tensor_sha256']
        or result['final_model_sha256']==result['initial_model_sha256']
        or result['all_diagnostic_samples_excluded_from_optimizer'] is not True or result['new_target_movies_opened']!=0
        or any(result[k] is not False for k in ('raw_detections_changed','source_selection_opened','authorized_for_submission'))):raise ValueError('Training identity or scope failed')
    for key,step in [('smoke',4),('checkpoint',800)]:
        row=result[key]
        if row['file']!=f'step-{step:04d}.pt' or row['exact_real_logit_reload'] is not True or sha(output/row['file'])!=row['sha256']:raise ValueError('Actual smoke/final checkpoint verification failed')
    counts=spec['supervision_counts']['diagnostic'];parent=counts['known_parent'];absent=counts['known_absent']
    for name in ('initial_diagnostic','physical_diagnostic','final_diagnostic'):
        row=result[name]
        if row['known_parent']!=parent or row['known_absent']!=absent or not 0<=row['correct_parent']<=parent or not 0<=row['correct_absent']<=absent:raise ValueError('Diagnostic inventory changed')
        if abs(row['nll']-row['loss_sum']/(parent+absent))>1e-12:raise ValueError('Micro average mismatch')
        if name in spec['initial_controls']:
            ref=spec['initial_controls'][name]
            if any(row[k]!=ref[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(row['nll']-ref['nll'])>2e-6:raise ValueError('Initial control replay changed')
    gate=diagnostic_gate(result['initial_diagnostic'],result['physical_diagnostic'],result['final_diagnostic'])
    if gate!=result['diagnostic_gate']:raise ValueError('Unchanged gate mismatch')
    return dict(status='verified_focus_null_balanced_head',run_id=RUN,notebook_sha256=sha(notebook),
        worker_result_sha256=sha(output/'result.json'),launcher=launcher,worker=result,diagnostic_gate=gate,
        eligible_for_source_tracking_evaluation=gate['passed'],authorized_for_submission=False,
        caveat='Training-domain feasibility only; diagnostic movies were in original encoder training; no independent or leaderboard claim.')


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite verified result')
    result=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}')
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
