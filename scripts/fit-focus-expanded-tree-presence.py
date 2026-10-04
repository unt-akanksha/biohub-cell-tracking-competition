"""Same fixed nonlinear presence method, twelve fitting movies only."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.focus_presence_trees import features,predict,SETTINGS
from research.focus_parent_presence import metrics
from research.focus_adaptation_training import diagnostic_gate
RUN='focus-expanded-tree-presence-v1'
PRIOR_SHA='ad1a546a0c903358c37ab20aee206cd0c617b093a1aa122254721f178ca70dd0'
ORIGINAL_TREE_SHA='ed2013cb85fc4111591fab357a7456163e59799c3313887d9c05d290061a2d02'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic();target=ROOT/f'reports/experiments/{RUN}-result.json';cache=ROOT/'.biohub/cache'/RUN
    if target.exists() or cache.exists():raise ValueError('Never overwrite experiment or fitted model')
    original_path=ROOT/'reports/experiments/focus-tree-presence-v1-result.json'
    if sha(original_path)!=ORIGINAL_TREE_SHA:raise ValueError('Exact previous nonlinear experiment required')
    original=json.loads(original_path.read_text())
    if original['settings']!=SETTINGS or any(sha(ROOT/p)!=v for p,v in original['source_hashes'].items()):
        raise ValueError('Original fixed nonlinear method changed')
    sources={p:sha(ROOT/p) for p in ('research/focus_presence_trees.py','scripts/train-focus-presence-trees.py',
        'scripts/fit-focus-expanded-tree-presence.py','reports/experiments/focus-expanded-tree-presence-v1-design.md')}
    prior=ROOT/'reports/experiments/focus-expanded-presence-v1-result.json'
    if sha(prior)!=PRIOR_SHA:raise ValueError('Actual completed expanded linear fit required')
    prior_result=json.loads(prior.read_text())
    if any(sha(ROOT/p)!=v for p,v in prior_result['source_hashes'].items()):raise ValueError('Verified expanded summary/metric implementation changed')
    data,evidence=runpy.run_path(str(ROOT/'scripts/fit-focus-expanded-presence.py'))['load']()
    if evidence!=prior_result['evidence']:raise ValueError('Exact twelve fitting/unchanged diagnostic evidence required')
    cache.mkdir();fitting=cache/'fitting.npz';model_path=cache/'model.json'
    np.savez_compressed(fitting,**data['fitting']);executable=Path(sys.base_prefix)/'python.exe'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    subprocess.run([str(executable),str(ROOT/'scripts/train-focus-presence-trees.py'),'--fitting',str(fitting),
        '--sha256',sha(fitting),'--output',str(model_path)],cwd=ROOT,env=env,timeout=120,check=True)
    model=json.loads(model_path.read_text(encoding='utf-8'))
    if (model['settings']!=SETTINGS or model['fitting_examples']!=len(data['fitting']['present'])
        or model['fitting_present']!=int(data['fitting']['present'].sum())):raise ValueError('Fixed fitting-only model contract changed')
    frozen_model=sha(model_path)
    diagnostic=metrics(dict(data['diagnostic'],offset=predict(model,features(data['diagnostic']))))
    fitting_metrics=metrics(dict(data['fitting'],offset=predict(model,features(data['fitting']))))
    gate=diagnostic_gate(evidence['baseline'],evidence['physical'],diagnostic)
    if sha(model_path)!=frozen_model or any(sha(ROOT/p)!=v for p,v in sources.items()):raise ValueError('Method/model changed during evaluation')
    result=dict(status='completed_twelve_movie_tree_presence_fit',run_id=RUN,settings=SETTINGS,
        fitting_metrics=fitting_metrics,diagnostic_metrics=diagnostic,diagnostic_gate=gate,model_sha256=frozen_model,
        fitting_file_sha256=sha(fitting),summary_result_sha256=sha(prior),original_tree_result_sha256=sha(original_path),source_hashes=sources,
        fitted_model_persisted_before_diagnostic=True,native_export_replay_max_abs=model['fitting_replay_max_abs'],
        sklearn_version=model['sklearn_version'],source_selection_opened=False,new_target_movies_opened=0,
        eligible_for_source_tracking_evaluation=gate['passed'],authorized_for_submission=False,gpu_seconds=0,
        elapsed_seconds=time.monotonic()-started,caveat='Same exposed training-domain diagnostic; not independent or a tracking score')
    target.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
