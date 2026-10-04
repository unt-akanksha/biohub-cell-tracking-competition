"""Verify actual frozen summaries, fit four movies, then evaluate the other four."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.focus_parent_presence import fit,metrics,FEATURES
from research.focus_adaptation_training import diagnostic_gate
RUN='focus-parent-presence-v1'
NOTEBOOK_SHA='d298b10947a73d96a693175dcca49937ba52567fe85678d66e4fc1e1fec6a8e1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    notebook=ROOT/'kaggle/biohub-focus-presence-summary-v1/biohub-focus-presence-summary-v1.ipynb'
    if sha(notebook)!=NOTEBOOK_SHA:raise ValueError('Exact immutable summary notebook required')
    nb=json.loads(notebook.read_text());node=next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value);spec=json.loads(runtime['training_spec.json']);policy=spec['contract']
    original=runpy.run_path(str(ROOT/'scripts/verify-focus-adaptation-head.py'))['verify'](ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-head-v1')
    work=ROOT/'.biohub/cache/kernel-outputs/focus-presence-summary-v1/focus_presence_summary';output=work/'outputs'
    for name,value in runtime.items():
        if (work/'runtime'/name).read_bytes()!=value.encode():raise ValueError('Actual summary runtime differs')
    if (ROOT/'research/focus_parent_presence.py').read_text(encoding='utf-8')!=runtime['focus_parent_presence.py']:raise ValueError('Staged presence policy changed before fitting')
    launcher=json.loads((work/'launcher_terminal.json').read_text());result=json.loads((output/'result.json').read_text())
    if (launcher['status']!='completed' or launcher['run_id']!='focus-presence-summary-v1' or launcher['declared_budget_seconds']!=3600
        or not 0<launcher['elapsed_seconds']<=3600 or launcher['submission_performed'] is not False):raise ValueError('Complete bounded launcher required')
    if (result['status']!='completed_focus_presence_summaries' or result['model_before']!=spec['model_tensor_sha256']
        or result['model_after']!=result['model_before'] or result['source_hashes']!=json.loads(runtime['source_hashes.json'])
        or result['training_spec_sha256']!=hashlib.sha256(runtime['training_spec.json'].encode()).hexdigest()
        or result['new_target_movies_opened']!=0 or any(result[k] is not False for k in ('optimizer_run','source_selection_opened','authorized_for_submission'))):
        raise ValueError('Frozen summary identity failed')
    expected=[(s,'fitting') for s in policy['fitting_stems']]+[(s,'diagnostic') for s in policy['diagnostic_stems']]
    if [(r['stem'],r['role']) for r in result['records']]!=expected:raise ValueError('Exact fixed training movie roles required')
    groups={r:[] for r in ('fitting','diagnostic')}
    for record in result['records']:
        stem=record['stem'];path=output/(stem+'.npz')
        if sha(path)!=record['sha256']:raise ValueError('Presence summary checksum mismatch')
        with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
        n=record['rows']
        if (set(arrays)!={'context','offset','present','conditional_nll','conditional_max','correct_if_present','target_indices','source_frame'}
            or arrays['context'].shape!=(n,len(FEATURES)) or any(not np.isfinite(v).all() for v in arrays.values())
            or any(v.shape!=(n,) for k,v in arrays.items() if k!='context')):raise ValueError('Finite complete presence schema required')
        labels=json.loads((ROOT/'.biohub/cache/focus-adaptation-labels-v1'/(stem+'.json')).read_text())
        ids=[];present=[];frames=[]
        for row in labels['rows']:
            for index,label in zip(row['target_indices'],row['labels']):
                if label>=0:
                    ids.append(index);present.append(int(label<row['null_index']));frames.append(row['source_frame'])
        if not all(np.array_equal(arrays[k],v) for k,v in [('target_indices',ids),('present',present),('source_frame',frames)]):
            raise ValueError('Presence labels/global target identities differ from frozen sparse audit')
        if not set(arrays['correct_if_present'])<={0,1} or ((arrays['present']==0)&(arrays['correct_if_present']!=0)).any():
            raise ValueError('Invalid parent correctness flags')
        replay=metrics(arrays);old=record['baseline']
        if any(replay[k]!=old[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(replay['nll']-old['nll'])>2e-6:
            raise ValueError('Actual per-movie baseline replay failed')
        groups[record['role']].append(arrays)
    pooled={role:{k:np.concatenate([d[k] for d in rows]) for k in rows[0]} for role,rows in groups.items()}
    baseline=metrics(pooled['diagnostic'])
    old=original['worker']['initial_diagnostic']
    if any(baseline[k]!=old[k] for k in ('known_parent','known_absent','correct_parent','correct_absent')) or abs(baseline['nll']-old['nll'])>2e-6:
        raise ValueError('Original full neural diagnostic not reproduced')
    if any(result[k]!=original['worker'][k] for k in ('initial_diagnostic','physical_diagnostic')):raise ValueError('GPU diagnostic replay changed')
    return pooled,dict(notebook_sha256=NOTEBOOK_SHA,worker_result_sha256=sha(output/'result.json'),launcher=launcher,
                      worker=result,baseline=baseline,physical=original['worker']['physical_diagnostic'])


def main():
    started=time.monotonic();target=ROOT/f'reports/experiments/{RUN}-result.json';fit_path=ROOT/f'reports/experiments/{RUN}-fit.json'
    if target.exists() or fit_path.exists():raise ValueError('Never overwrite fitted parameters or evaluated experiment')
    frozen={p:sha(ROOT/p) for p in ('research/focus_parent_presence.py','scripts/fit-focus-parent-presence.py','reports/experiments/focus-parent-presence-v1-design.md')}
    data,evidence=load();model=fit(data['fitting'],'fitting')
    fit_path.write_text(json.dumps(dict(model=model,source_hashes=frozen,summary_worker_sha256=evidence['worker_result_sha256']),indent=2,allow_nan=False))
    # Persisted fitting parameters precede the only corrected diagnostic evaluation.
    fitted=metrics(data['fitting'],model);final=metrics(data['diagnostic'],model)
    gate=diagnostic_gate(evidence['baseline'],evidence['physical'],final)
    if any(sha(ROOT/p)!=v for p,v in frozen.items()):raise ValueError('Declared method changed during evaluation')
    result=dict(status='completed_training_only_parent_presence_fit',run_id=RUN,evidence=evidence,
        fitting_metrics=fitted,diagnostic_metrics=final,diagnostic_gate=gate,fit_sha256=sha(fit_path),source_hashes=frozen,
        coefficients_persisted_before_diagnostic=True,source_selection_opened=False,new_target_movies_opened=0,
        eligible_for_source_tracking_evaluation=gate['passed'],authorized_for_submission=False,gpu_fit_seconds=0,
        elapsed_seconds=time.monotonic()-started,caveat='Training-domain diagnostic only; no independent or leaderboard score')
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
