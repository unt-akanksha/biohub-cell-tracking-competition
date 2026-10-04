from copy import deepcopy
from pathlib import Path
import ast
import runpy
import pytest
from research.blindspot_functional_audit import assess

ROOT=Path(__file__).resolve().parents[1]


def records():
    return [dict(center_gradient_abs=0.,center_output_delta=0.,gradient_l1=0.) for _ in range(5)]


def test_inactive_local_relu_does_not_imply_global_constant():
    result=assess(records(),.1)
    assert result['passed'] and result['inactive_local_probes']==5


def test_globally_constant_model_is_rejected():
    assert not assess(records(),0.)['passed']


@pytest.mark.parametrize('key',['center_gradient_abs','center_output_delta'])
def test_center_leakage_still_rejected(key):
    r=records(); r[0][key]=.001
    assert not assess(r,.1)['passed']


def test_nonfinite_and_insufficient_evidence_rejected():
    with pytest.raises(ValueError): assess(records(),float('nan'))
    with pytest.raises(ValueError): assess(records()[:1],.1)
    r=records();r[0]['gradient_l1']=float('inf')
    with pytest.raises(ValueError): assess(r,.1)


def test_repair_freezes_model_scope_and_saves_before_verification():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-blindspot-real-probe-v2.py'))['build']()
    def bundle(path_or_nb):
        if isinstance(path_or_nb,Path):
            import json
            path_or_nb=json.loads(path_or_nb.read_text())
        source=''.join(path_or_nb['cells'][1]['source'])
        return ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
                               and getattr(n.targets[0],'id',None)=='runtime_sources'))
    old=bundle(ROOT/'kaggle/biohub-blindspot-real-probe-v1/biohub-blindspot-real-probe-v1.ipynb')
    new=bundle(nb)
    for key in ('blindspot_restoration.py','blindspot_training_contract.py','split.json','cpu_probe.json'):
        assert new[key]==old[key]
    worker=new['run_pilot.py']
    assert worker.index("checkpoint=args.output/'last.pt'")<worker.index("diagnostic_snapshot.json")<worker.index('sensitivity=audit')
    assert 'optimizer=optimizer.state_dict()' in worker and 'torch_rng_cuda=' in worker
    assert 'range(100)' in worker and "lr=.001" in worker
    assert not meta['enable_internet'] and meta['enable_gpu']
