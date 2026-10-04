"""Verify actual smoke runtime, stress identities, gradients and checkpoint."""
import ast
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_parent_dropout import augment
RUN='focus-parent-dropout-smoke-v3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    staged=ROOT/'kaggle'/('biohub-'+RUN);identity=json.loads((staged/'staged_identity.json').read_text())
    path=staged/('biohub-'+RUN+'.ipynb');builder=ROOT/'scripts/build-focus-parent-dropout-smoke-v3.py'
    if sha(path)!=identity['notebook_sha256'] or sha(builder)!=identity['builder_sha256'] or sha(staged/'kernel-metadata.json')!=identity['metadata_sha256']:raise ValueError('Frozen stage changed')
    nb=json.loads(path.read_text());source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=runpy.run_path(str(builder))['decode_runtime'](source);spec=json.loads(runtime['training_spec.json']);audit=json.loads(runtime['dropout_audit.json'])
    work=ROOT/f'.biohub/cache/kernel-outputs/{RUN}/focus_parent_dropout_smoke_v3'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual runtime differs from launch')
    if spec!=runpy.run_path(str(builder))['make_spec']():raise ValueError('Upstream audited scope changed')
    launcher=json.loads((work/'launcher_terminal.json').read_text());output=work/'outputs';result=json.loads((output/'result.json').read_text())
    if launcher['status']!='completed' or launcher['run_id']!=RUN or launcher['declared_budget_seconds']!=900 or not 0<launcher['elapsed_seconds']<=900 or launcher['submission_performed'] is not False:raise ValueError('Bounded terminal smoke required')
    if (result['status']!='completed_focus_parent_dropout_smoke' or result['settings']!=spec['settings']
        or result['null_weight']!=spec['null_weight'] or result['supervision_counts']!=spec['supervision_counts']
        or result['source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['initial_model_sha256']!=spec['model_tensor_sha256'] or result['final_model_sha256']==result['initial_model_sha256']
        or result['frozen_before']!=result['frozen_after'] or result['final_diagnostic'] is not None
        or result['diagnostic_gate']!={'passed':False,'scope':'Four-step functionality only; no quality evaluation'}
        or result['verified_augmentation_pairs']!=audit['eligible_pairs'] or len(result['updates'])!=4):raise ValueError('Fixed smoke identity/scope changed')
    for key in ('physical_diagnostic','initial_diagnostic'):
        actual=result[key];old=spec['initial_controls'][key]
        if any(actual[k]!=old[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(actual['nll']-old['nll'])>2e-6:raise ValueError('Original complete control replay failed')
    roots=[ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1/focus_adaptation_features/outputs',ROOT/'.biohub/cache/kernel-outputs/focus-extra-fit-features-v1/focus_extra_fit_features/outputs']
    entries=[];lookup={(r['stem'],r['source_frame']):r for r in audit['records']}
    for group,root in zip(spec['feature_groups'],roots):
        for r in group['feature_records']:
            if r['role']!='fitting':continue
            manifest=json.loads((root/r['stem']/'manifest.json').read_text())
            for pair in manifest['pairs']:
                key=(r['stem'],pair['source_frame'])
                if key in lookup and lookup[key]['provenance']['eligible_nonempty_source']:
                    entries.append((r['stem'],pair,root/r['stem']/pair['file'],lookup[key]))
    if len(entries)!=audit['eligible_pairs']:raise ValueError('Complete eligible fitting inventory required')
    biggest=max(entries,key=lambda e:e[1]['source_nodes']*e[1]['target_nodes'])
    most=max(entries,key=lambda e:len(e[3]['provenance']['synthetic_null_columns']))
    for step,(entry,augmented),actual in zip(range(1,5),[(biggest,False),(biggest,True),(most,False),(most,True)],result['updates']):
        stem,pair,path,row=entry
        if sha(path)!=pair['sha256']:raise ValueError('Stress packet changed')
        with np.load(path,allow_pickle=False) as data:packet={k:data[k].copy() for k in data.files}
        altered=augment(dict(packet=packet,stem=stem,role='fitting'))
        if altered['provenance']!=row['provenance']:raise ValueError('Actual local dropout replay differs')
        expected=dict(step=step,stem=stem,source_frame=pair['source_frame'],pair_sha256=pair['sha256'],augmentation=altered['provenance'] if augmented else None)
        if any(actual[k]!=v for k,v in expected.items()) or not all(math.isfinite(actual[k]) for k in ('loss','gradient_norm')) or actual['gradient_norm']<=0:raise ValueError('Exact finite original/augmented stress schedule required')
    checkpoint=result['checkpoint']
    if checkpoint!=result['smoke'] or checkpoint['file']!='step-0004.pt' or checkpoint['exact_real_logit_reload'] is not True or sha(output/checkpoint['file'])!=checkpoint['sha256']:raise ValueError('Actual exact-reload checkpoint required')
    if not 0<result['peak_allocated_bytes']<16*1024**3:raise ValueError('Bounded nonzero GPU memory required')
    if result['raw_detections_changed'] is not False or result['source_selection_opened'] is not False or result['new_target_movies_opened']!=0 or result['authorized_for_submission'] is not False:raise ValueError('No inference/source/target mutation allowed')
    return dict(status='verified_parent_dropout_smoke',worker_result_sha256=sha(output/'result.json'),launcher=launcher,worker=result,
        eligible_for_bounded_training=True,authorized_for_submission=False)


if __name__=='__main__':
    target=ROOT/f'reports/experiments/{RUN}-result.json'
    if target.exists():raise ValueError('Never overwrite actual verification')
    result=verify();target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='worker'},indent=2))
