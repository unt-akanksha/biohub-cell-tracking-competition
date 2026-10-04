import ast
import hashlib
from pathlib import Path
import runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_fixed_mixture_math_without_torch():
    a=np.linspace(-10,10,101); b=a[::-1]*.7
    def mix(x,y):
        return np.logaddexp(-np.logaddexp(0,-x),-np.logaddexp(0,-y))-np.logaddexp(-np.logaddexp(0,x),-np.logaddexp(0,y))
    np.testing.assert_allclose(1/(1+np.exp(-mix(a,b))), (1/(1+np.exp(-a))+1/(1+np.exp(-b)))/2)
    extreme=np.array([-1000.,-100.,0.,100.,1000.])
    np.testing.assert_allclose(mix(extreme,extreme),extreme)


def test_immutable_bounded_ensemble_bundle():
    paths=list((ROOT/'kaggle').glob('biohub-owned-detector-*-v1/*.ipynb'))
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-owned-detector-ensemble-probe.py'))['build']()
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert meta['kernel_sources'][-1]=='indarkarhana/biohub-owned-detector-fit-pair-v1/1'
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    assert not nb['metadata']['codex']['target_audit_opened']
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(node.value)
    for name in ('owned_detector_ensemble','owned_detector_pu','owned_detector_logit_targets'):
        assert runtime[name+'.py']==(ROOT/f'research/{name}.py').read_text()
    for name,code in runtime.items():
        if name.endswith('.py'): ast.parse(code)
    assert hashlib.sha256(runtime['owned_probe.json'].encode()).hexdigest()=='889a94d0a7bb60d40bde3a42a1343072a33cc68a0680dd0d083c41814efaf2ba'
    launch=''.join(nb['cells'][-1]['source'])
    assert launch.index("'-m','pytest'")<launch.index('process = subprocess.Popen')
    assert "'--secondary', str(secondary)" in launch
    assert "'--checkpoint', str(checkpoint)" in launch
    assert 'test_owned_detector_ensemble.py' in ' '.join(runtime)
