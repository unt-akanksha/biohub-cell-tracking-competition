"""Verify all saved joint checkpoints, fitting queue and unchanged diagnostics."""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from focus_joint_training import SETTINGS
from research.focus_adaptation_training import diagnostic_gate
RUN='focus-joint-training-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    staged=ROOT/f'kaggle/biohub-{RUN}';identity=json.loads((staged/'staged_identity.json').read_text());notebook=staged/f'biohub-{RUN}.ipynb'
    builder=ROOT/'scripts/build-focus-joint-training.py'
    if sha(notebook)!=identity['notebook_sha256'] or sha(builder)!=identity['builder_sha256'] or sha(staged/'kernel-metadata.json')!=identity['metadata_sha256']:raise ValueError('Frozen launch identity changed')
    nb=json.loads(notebook.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);spec=json.loads(runtime['training_spec.json'])
    if spec!=runpy.run_path(str(builder))['make_spec']():raise ValueError('Verified smoke/upstream specification changed')
    work=folder/'focus_joint_training';output=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual training runtime changed')
    launcher=json.loads((work/'launcher_terminal.json').read_text());result=json.loads((output/'result.json').read_text())
    if launcher['status']!='completed' or launcher['run_id']!=RUN or launcher['declared_budget_seconds']!=1800 or not 0<launcher['elapsed_seconds']<=1800 or launcher['submission_performed'] is not False:raise ValueError('Actual bounded completed launcher required')
    if (result['status']!='completed_focus_joint_training' or result['settings']!=SETTINGS or len(result['updates'])!=800
        or result['source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['initial_model_sha256']!=spec['model_tensor_sha256'] or result['final_model_sha256']==result['initial_model_sha256']
        or result['null_weight']!=spec['null_weight'] or result['all_diagnostic_samples_excluded_from_optimizer'] is not True
        or result['source_selection_opened'] is not False or result['new_target_movies_opened']!=0 or result['authorized_for_submission'] is not False):raise ValueError('Complete fixed joint-training identity/scope required')
    for name in ('encoder','head'):
        if result[name+'_before']==result[name+'_after']:raise ValueError('Both trained groups must change')
    for name in ('detector','flow'):
        if result[name+'_before']!=result[name+'_after']:raise ValueError('Frozen component changed')
    expected_steps=[4]+list(range(100,801,100))
    if [r['step'] for r in result['checkpoints']]!=expected_steps:raise ValueError('All planned checkpoint records required')
    for row in result['checkpoints']:
        if row['file']!=f"step-{row['step']:04d}.pt" or row['exact_real_image_reload'] is not True or sha(output/row['file'])!=row['sha256']:raise ValueError('Actual checkpoint file/reload proof changed')
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    fitting=[];diagnostic=[];manifest_rows=[]
    for group,root in zip(spec['feature_groups'],roots):
        for r in group['feature_records']:
            if r['role']=='replay':continue
            manifest=json.loads((root/r['stem']/'manifest.json').read_text())
            for pair in manifest['pairs']:
                if not pair['known_parent']+pair['known_absent']:continue
                key=(r['stem'],pair['source_frame'])
                if r['role']=='diagnostic':diagnostic.append(key)
                elif pair['source_nodes']>0:fitting.append(key);manifest_rows.append(dict(pair,stem=r['stem']))
    if result['fitting_pairs']!=len(fitting) or result['diagnostic_pairs']!=len(diagnostic):raise ValueError('Full supervised inventory changed')
    rng=random.Random(SETTINGS['seed']);queue=[];scale=SETTINGS['amp_initial_scale'];skips=0
    for step,row in enumerate(result['updates'],1):
        if not queue:queue=list(range(len(fitting)));rng.shuffle(queue)
        expected=fitting[queue.pop()]
        if row['step']!=step or (row['stem'],row['source_frame'])!=expected or row['nonzero_gradients']!={'encoder':True,'head':True} or not all(math.isfinite(row[k]) for k in ('loss','gradient_norm')):raise ValueError('Declared finite fitting-only queue/updates changed')
        if not 1<=row['attempts']<=SETTINGS['max_amp_attempts'] or len(row['overflow_retries'])!=row['attempts']-1:raise ValueError('Bounded retry trace required')
        for i,retry in enumerate(row['overflow_retries'],1):
            if retry['attempt']!=i or retry['scale_before']!=scale or retry['scale_after']!=scale*.5 or retry['optimizer_skipped'] is not True or retry['parameters_unchanged'] is not True:raise ValueError('Overflow guard/backoff trace changed')
            scale=retry['scale_after'];skips+=1
        if row['successful_scale']!=scale:raise ValueError('Actual successful scaler differs')
    picks=[manifest_rows[0],max(manifest_rows,key=lambda p:p['source_nodes']*p['target_nodes']),max(manifest_rows,key=lambda p:p['known_absent']),max(manifest_rows,key=lambda p:p['known_parent'])]+manifest_rows
    chosen=[];seen=set()
    for p in picks:
        key=(p['stem'],p['source_frame'])
        if key not in seen:chosen.append(p);seen.add(key)
        if len(chosen)==4:break
    expected=[dict(stem=p['stem'],source_frame=p['source_frame'],pair_sha256=p['sha256'],exact_logit_replay=True) for p in chosen]
    if result['preflight']!=expected:raise ValueError('Four exact initial image replays required')
    image_keys={(r['stem'],r['source_frame']) for r in result['normalized_images']}
    required=set(diagnostic)|{(r['stem'],r['source_frame']) for r in result['updates']}|seen
    if image_keys!=required or len(image_keys)!=len(result['normalized_images']):raise ValueError('Exact fitting/diagnostic image coverage required')
    parent=spec['supervision_counts']['diagnostic']['known_parent'];absent=spec['supervision_counts']['diagnostic']['known_absent']
    for name in ('physical_diagnostic','initial_diagnostic','final_diagnostic'):
        row=result[name]
        if row['known_parent']!=parent or row['known_absent']!=absent or not 0<=row['correct_parent']<=parent or not 0<=row['correct_absent']<=absent or abs(row['nll']-row['loss_sum']/(parent+absent))>1e-12:raise ValueError('Complete micro-averaged diagnostic counts changed')
        if name in spec['initial_controls']:
            old=spec['initial_controls'][name]
            if any(row[k]!=old[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(row['nll']-old['nll'])>2e-6:raise ValueError('Original control replay failed')
    gate=diagnostic_gate(result['initial_diagnostic'],result['physical_diagnostic'],result['final_diagnostic'])
    if gate!=result['diagnostic_gate']:raise ValueError('Unchanged diagnostic gate differs')
    return dict(status='verified_focus_joint_training',run_id=RUN,notebook_sha256=sha(notebook),worker_result_sha256=sha(output/'result.json'),launcher=launcher,
        worker=result,diagnostic_gate=gate,verified_overflow_skips=skips,exact_fitting_queue_replayed=True,
        eligible_for_source_tracking_evaluation=gate['passed'],authorized_for_submission=False,
        caveat='Original training-domain diagnostics only; not independent whole-system or leaderboard evidence.')


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite actual verification')
    result=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}');target.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='worker'},indent=2))
