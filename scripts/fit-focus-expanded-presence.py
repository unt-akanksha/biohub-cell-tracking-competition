"""Verify twelve fitting summaries, persist fixed fit, test unchanged diagnostics."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_parent_presence import fit,metrics
from research.focus_summary_validation import validate_summary
from research.focus_adaptation_training import diagnostic_gate
RUN='focus-expanded-presence-v1';SUMMARY='focus-expanded-summary-v1'
METHOD_SHA='9c6c473adf8f2f4b595a71e94ea6009cf0a006bd0573021fcecefa6b71151539'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def load():
    if sha(ROOT/'research/focus_parent_presence.py')!=METHOD_SHA:raise ValueError('Original fixed linear method must remain unchanged')
    stage=ROOT/'kaggle'/('biohub-'+SUMMARY);identity=json.loads((stage/'staged_identity.json').read_text())
    notebook=stage/('biohub-'+SUMMARY+'.ipynb')
    if (identity['status']!='staged_not_launched' or identity['run_id']!=SUMMARY
        or sha(notebook)!=identity['notebook_sha256'] or sha(stage/'kernel-metadata.json')!=identity['metadata_sha256']
        or sha(ROOT/'scripts/build-focus-expanded-summary.py')!=identity['builder_sha256']):
        raise ValueError('Exact immutable prelaunch summary identity required')
    nb=json.loads(notebook.read_text())
    node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);spec=json.loads(runtime['training_spec.json'])
    if spec!=runpy.run_path(str(ROOT/'scripts/build-focus-expanded-summary.py'))['make_spec']():
        raise ValueError('Expanded fitting specification differs from actual upstream evidence')
    work=ROOT/'.biohub/cache/kernel-outputs'/SUMMARY/'focus_expanded_summary';output=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual expanded summary runtime changed')
    launcher=json.loads((work/'launcher_terminal.json').read_text());result=json.loads((output/'result.json').read_text())
    if (launcher['status']!='completed' or launcher['run_id']!=SUMMARY or launcher['declared_budget_seconds']!=3600
        or not 0<launcher['elapsed_seconds']<=3600 or launcher['submission_performed'] is not False):
        raise ValueError('Complete bounded summary launcher required')
    if (result['status']!='completed_focus_expanded_fitting_summaries'
        or result['model_before']!=spec['model_tensor_sha256'] or result['model_after']!=result['model_before']
        or result['source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['replayed_stems']!=list(spec['replay_summaries']) or result['new_target_movies_opened']!=0
        or any(result[k] is not False for k in ('optimizer_run','diagnostic_evaluated','source_selection_opened','authorized_for_submission'))):
        raise ValueError('Frozen original model or fitting-only summary terminal differs')
    expected=[(s,'fitting') for s in spec['contract']['fitting_stems']]
    if len(expected)!=12 or len(set(expected))!=12 or [(r['stem'],r['role']) for r in result['records']]!=expected:
        raise ValueError('Exact twelve unique fitting movies required')
    old_data,old=runpy.run_path(str(ROOT/'scripts/fit-focus-parent-presence.py'))['load']()
    if (old['worker_result_sha256']!=spec['previous_summary_worker_sha256']
        or {r['stem']:r['sha256'] for r in old['worker']['records'] if r['role']=='diagnostic'}!=spec['unchanged_diagnostic_summaries']):
        raise ValueError('Original diagnostic summary identity changed')
    old_output=ROOT/'.biohub/cache/kernel-outputs/focus-presence-summary-v1/focus_presence_summary/outputs'
    rows=[];records=[]
    for record in result['records']:
        stem=record['stem'];path=output/(stem+'.npz')
        if sha(path)!=record['sha256']:raise ValueError('Expanded fitting summary checksum mismatch')
        data=arrays(path);previous=stem in spec['replay_summaries']
        label_dir='focus-adaptation-labels-v1' if previous else 'focus-extra-fit-labels-v1'
        label_path=ROOT/'.biohub/cache'/label_dir/(stem+'.json')
        labels=json.loads(label_path.read_text());baseline=validate_summary(data,record,labels)
        if previous:
            reference=old_output/(stem+'.npz')
            if sha(reference)!=spec['replay_summaries'][stem]:raise ValueError('Original fitting summary changed')
            prior=arrays(reference)
            if set(data)!=set(prior) or any(not np.array_equal(data[k],prior[k]) for k in data):
                raise ValueError('Actual original fitting array replay failed')
        rows.append(data);records.append(dict(record,labels_sha256=sha(label_path),host_baseline=baseline))
    fitting={k:np.concatenate([d[k] for d in rows]) for k in rows[0]}
    return dict(fitting=fitting,diagnostic=old_data['diagnostic']),dict(notebook_sha256=identity['notebook_sha256'],
        staged_identity_sha256=sha(stage/'staged_identity.json'),worker_result_sha256=sha(output/'result.json'),
        launcher=launcher,records=records,contract=spec['contract'],baseline=old['baseline'],physical=old['physical'],
        unchanged_diagnostic_summaries=spec['unchanged_diagnostic_summaries'],previous_summary_worker_sha256=old['worker_result_sha256'])


def execute_fit(data,evidence,fit_path,source_hashes=None):
    if fit_path.exists():raise ValueError('Never overwrite fitted parameters')
    model=fit(data['fitting'],'fitting')
    fit_path.write_text(json.dumps(dict(model=model,summary_worker_sha256=evidence['worker_result_sha256'],
        unchanged_diagnostic_summaries=evidence['unchanged_diagnostic_summaries'],
        source_hashes=source_hashes or {}),indent=2,allow_nan=False))
    # First corrected diagnostic evaluation is strictly after durable parameter export.
    persisted=json.loads(fit_path.read_text())['model']
    if persisted!=model:raise ValueError('Saved expanded-data parameters differ')
    fitted=metrics(data['fitting'],persisted);final=metrics(data['diagnostic'],persisted)
    return fitted,final,diagnostic_gate(evidence['baseline'],evidence['physical'],final)


def main():
    started=time.monotonic();target=ROOT/f'reports/experiments/{RUN}-result.json';fit_path=ROOT/f'reports/experiments/{RUN}-fit.json'
    if target.exists() or fit_path.exists():raise ValueError('Never overwrite fitted/evaluated experiment')
    sources=('research/focus_parent_presence.py','research/focus_summary_validation.py','research/focus_adaptation_training.py',
        'scripts/fit-focus-expanded-presence.py','reports/experiments/focus-expanded-presence-v1-protocol.md')
    frozen={p:sha(ROOT/p) for p in sources};data,evidence=load()
    fitted,final,gate=execute_fit(data,evidence,fit_path,frozen)
    if any(sha(ROOT/p)!=v for p,v in frozen.items()):raise ValueError('Declared fitting method changed during evaluation')
    result=dict(status='completed_twelve_movie_parent_presence_fit',run_id=RUN,evidence=evidence,source_hashes=frozen,
        fitting_metrics=fitted,diagnostic_metrics=final,diagnostic_gate=gate,fit_sha256=sha(fit_path),
        coefficients_persisted_before_diagnostic=True,source_selection_opened=False,new_target_movies_opened=0,
        eligible_for_source_tracking_evaluation=gate['passed'],authorized_for_submission=False,gpu_fit_seconds=0,
        elapsed_seconds=time.monotonic()-started,caveat='Same exposed training-domain diagnostic; not an independent tracking score')
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
