"""Verify exact staged/runtime identity, selected stress pairs and checkpoint."""
import ast
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_joint_probe import SETTINGS
RUN='focus-joint-smoke-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    staged=ROOT/f'kaggle/biohub-{RUN}';identity=json.loads((staged/'staged_identity.json').read_text())
    notebook=staged/f'biohub-{RUN}.ipynb';builder=ROOT/'scripts/build-focus-joint-smoke.py'
    if sha(notebook)!=identity['notebook_sha256'] or sha(builder)!=identity['builder_sha256'] or sha(staged/'kernel-metadata.json')!=identity['metadata_sha256']:raise ValueError('Frozen launch identity changed')
    nb=json.loads(notebook.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);spec=json.loads(runtime['training_spec.json'])
    if spec!=runpy.run_path(str(builder))['make_spec']():raise ValueError('Actual verified training input identity changed')
    work=folder/'focus_joint_smoke';output=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual runtime changed')
    launcher=json.loads((work/'launcher_terminal.json').read_text());result=json.loads((output/'result.json').read_text())
    if launcher['status']!='completed' or launcher['run_id']!=RUN or launcher['declared_budget_seconds']!=3600 or not 0<launcher['elapsed_seconds']<=3600 or launcher['submission_performed'] is not False:raise ValueError('Completed bounded launcher required')
    if (result['status']!='completed_focus_joint_smoke' or result['settings']!=SETTINGS or len(result['steps'])!=4
        or result['source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['initial_model_sha256']!=spec['model_tensor_sha256'] or result['initial_model_sha256']==result['final_model_sha256']
        or any(result[k] is not False for k in ('diagnostic_evaluated','source_selection_opened','authorized_for_submission'))
        or result['new_target_movies_opened']!=0 or result['null_weight']!=spec['null_weight']):raise ValueError('Actual joint scope/identity failed')
    for group in ('encoder','head'):
        if result[group+'_before']==result[group+'_after']:raise ValueError('Both trainable groups must change')
    for group in ('detector','flow'):
        if result[group+'_before']!=result[group+'_after']:raise ValueError('Frozen external component changed')
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    available=[];diagnostic_count=0
    for group,root in zip(spec['feature_groups'],roots):
        for r in group['feature_records']:
            if r['role']=='replay':continue
            manifest=json.loads((root/r['stem']/'manifest.json').read_text())
            for p in manifest['pairs']:
                if not p['known_parent']+p['known_absent']:continue
                if r['role']=='diagnostic':diagnostic_count+=1;continue
                if p['source_nodes']>0:available.append(dict(p,stem=r['stem']))
    picks=[available[0],max(available,key=lambda p:p['source_nodes']*p['target_nodes']),max(available,key=lambda p:p['known_absent']),max(available,key=lambda p:p['known_parent'])]+available
    selected=[];seen=set()
    for p in picks:
        key=(p['stem'],p['source_frame'])
        if key not in seen:selected.append(p);seen.add(key)
        if len(selected)==4:break
    expected=[dict(stem=p['stem'],source_frame=p['source_frame'],pair_sha256=p['sha256'],source_nodes=p['source_nodes'],target_nodes=p['target_nodes'],exact_feature_and_logit_replay=True) for p in selected]
    if result['replays']!=expected or result['fitting_pairs']!=len(available) or result['diagnostic_pairs']!=diagnostic_count:raise ValueError('Actual stress-pair inventory changed')
    for i,(row,pair) in enumerate(zip(result['steps'],expected),1):
        if row['step']!=i or (row['stem'],row['source_frame'])!=(pair['stem'],pair['source_frame']) or row['nonzero_gradients']!={'encoder':True,'head':True} or not math.isfinite(row['loss']) or not math.isfinite(row['gradient_norm']):raise ValueError('Four finite joint-gradient updates required')
    if {(r['stem'],r['source_frame']) for r in result['normalized_images']}!={(p['stem'],p['source_frame']) for p in expected}:raise ValueError('Unexpected image scope')
    checkpoint=result['checkpoint']
    if checkpoint['file']!='step-0004.pt' or checkpoint['exact_real_image_reload'] is not True or sha(output/checkpoint['file'])!=checkpoint['sha256']:raise ValueError('Actual saved real-image replay checkpoint required')
    if not 0<result['peak_allocated_bytes']<=result['peak_reserved_bytes']<=result['gpu_total_bytes']:raise ValueError('Actual bounded memory measurements required')
    return dict(status='verified_focus_joint_smoke',notebook_sha256=sha(notebook),worker_result_sha256=sha(output/'result.json'),launcher=launcher,worker=result,
        eligible_for_bounded_joint_training=True,authorized_for_submission=False,caveat='Four fitting updates verify functionality/resources only, not quality or full-training runtime.')


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite verified result')
    receipt=verify(ROOT/f'.biohub/cache/kernel-outputs/{RUN}');target.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
