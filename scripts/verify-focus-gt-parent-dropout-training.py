"""Actual 800-step training verification, including both complete sample queues."""
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_adaptation_training import diagnostic_gate
RUN='focus-gt-parent-dropout-training-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    staged=ROOT/'kaggle'/('biohub-'+RUN);nbpath=staged/('biohub-'+RUN+'.ipynb')
    builder=ROOT/'scripts/build-focus-gt-parent-dropout-training.py';identity=json.loads((staged/'staged_identity.json').read_text())
    if sha(nbpath)!=identity['notebook_sha256'] or sha(builder)!=identity['builder_sha256'] or sha(staged/'kernel-metadata.json')!=identity['metadata_sha256']:raise ValueError('Frozen launch changed')
    build=runpy.run_path(str(builder));nb=json.loads(nbpath.read_text());runtime=build['decode_runtime'](''.join(nb['cells'][1]['source']))
    spec=json.loads(runtime['training_spec.json']);audit=json.loads(runtime['dropout_audit.json'])
    if spec!=build['make_spec']():raise ValueError('Actual upstream smoke and training scope changed')
    work=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/focus_gt_parent_dropout_training';out=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual executed runtime changed')
    result=json.loads((out/'result.json').read_text());launcher=json.loads((work/'launcher_terminal.json').read_text())
    if launcher['status']!='completed' or launcher['run_id']!=RUN or launcher['declared_budget_seconds']!=900 or not 0<launcher['elapsed_seconds']<=900 or launcher['submission_performed'] is not False:raise ValueError('Bounded complete launcher required')
    if (result['status']!='completed_focus_gt_parent_dropout_training' or result['settings']!=spec['settings'] or len(result['updates'])!=800
        or result['null_weight']!=spec['null_weight'] or result['supervision_counts']!=spec['supervision_counts']
        or result['source_hashes']!=json.loads(runtime['source_hashes.json']) or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['initial_model_sha256']!=spec['model_tensor_sha256'] or result['final_model_sha256']==result['initial_model_sha256']
        or result['frozen_before']!=result['frozen_after'] or result['frozen_before']!='a9a07f3362fd2891894ee3b534ab34a9395d68cbce7e2e9ac2be16a144ac5ef1'
        or result['verified_augmentation_pairs']!=1121 or result['fitting_pairs']!=1121 or result['diagnostic_pairs']!=377):raise ValueError('Fixed training/model/data identity changed')
    expected_steps=[4]+list(range(100,801,100))
    if [r['step'] for r in result['checkpoints']]!=expected_steps:raise ValueError('All nine planned checkpoints required')
    for record in result['checkpoints']:
        cp=record['checkpoint']
        if cp['file']!=f"step-{record['step']:04d}.pt" or cp['exact_real_logit_reload'] is not True or sha(out/cp['file'])!=cp['sha256']:raise ValueError('Actual checkpoint/reload proof changed')
    if result['smoke']!=result['checkpoints'][0]['checkpoint'] or result['checkpoint']!=result['checkpoints'][-1]['checkpoint']:raise ValueError('Fixed final800 checkpoint required')
    records=audit['records']
    if len(records)!=1121 or any(not r['provenance']['eligible_nonempty_source'] for r in records):raise ValueError('Exact original and augmented pairing required')
    queues={'original':[],'augmented':[]};rng=random.Random(spec['settings']['seed'])
    for step,actual in enumerate(result['updates'],1):
        stream='original' if step%2 else 'augmented'
        if not queues[stream]:queues[stream]=list(range(1121));rng.shuffle(queues[stream])
        row=records[queues[stream].pop()]
        expected=dict(step=step,stem=row['stem'],source_frame=row['source_frame'],pair_sha256=row['packet_sha256'],augmentation=row['provenance'] if stream=='augmented' else None)
        if any(actual[k]!=v for k,v in expected.items()) or not all(math.isfinite(actual[k]) for k in ('loss','gradient_norm')) or actual['gradient_norm']<=0:raise ValueError('Exact finite alternating fitting schedule required')
    for key in ('physical_diagnostic','initial_diagnostic','final_diagnostic'):
        r=result[key]
        if r['known_parent']!=2645 or r['known_absent']!=27 or not 0<=r['correct_parent']<=2645 or not 0<=r['correct_absent']<=27 or not math.isfinite(r['nll']) or abs(r['loss_sum']/2672-r['nll'])>1e-12:raise ValueError('Complete unweighted diagnostic required')
        if key in spec['initial_controls']:
            old=spec['initial_controls'][key]
            if any(r[k]!=old[k] for k in ('correct_parent','correct_absent')) or abs(r['nll']-old['nll'])>2e-6:raise ValueError('Original control replay changed')
    gate=diagnostic_gate(result['initial_diagnostic'],result['physical_diagnostic'],result['final_diagnostic'])
    if result['diagnostic_gate']!=gate:raise ValueError('Unchanged real-data gate differs')
    if result['all_diagnostic_samples_excluded_from_optimizer'] is not True or result['raw_detections_changed'] is not False or result['source_selection_opened'] is not False or result['new_target_movies_opened']!=0 or result['authorized_for_submission'] is not False:raise ValueError('Training-only scope required')
    return dict(status='verified_parent_dropout_training',launcher=launcher,worker_result_sha256=sha(out/'result.json'),worker=result,
        exact_alternating_sampling_replayed=True,diagnostic_gate=gate,eligible_for_source_tracking_evaluation=gate['passed'],authorized_for_submission=False)


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite actual verification')
    result=verify();target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='worker'},indent=2))
