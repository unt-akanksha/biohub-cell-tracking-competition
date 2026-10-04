import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
from research.focus_extra_feature_scope import scope

ROOT=Path(__file__).resolve().parents[1]


def assignment(nb):
    return ast.literal_eval(next(n.value for n in ast.parse(''.join(nb['cells'][1]['source'])).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources'))


def test_same_collector_and_new_fitting_only_scope():
    policy=scope((ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    assert len(policy['fitting_stems'])==8 and policy['diagnostic_stems']==[] and len(policy['unchanged_diagnostic_stems'])==4
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-focus-extra-fit-features.py'))['build']({'contract':policy,'movies':[]})
    old=json.loads((ROOT/'kaggle/biohub-focus-adaptation-features-v1/biohub-focus-adaptation-features-v1.ipynb').read_text())
    before=assignment(old);after=assignment(nb)
    for name in ('run_pilot.py','focus_cached_pair.py','focus_feature_spec.py','image_motion_residual.py','raw_centroid_flow_sampling.py'):
        assert before[name]==after[name]
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu'] and meta['is_private']
    assert meta['kernel_sources'][1]=='indarkarhana/biohub-focus-extra-fit-cache-v1/1'
    last=''.join(nb['cells'][-1]['source'])
    assert "kernel_root('biohub-focus-extra-fit-cache-v1','')" in last and '3480-' in last
    assert 'ZIP_STORED' in last and 'focus_extra_fit_features_bundle.zip' in last
    for c in nb['cells']:ast.parse(''.join(c['source']))
    for k,v in after.items():
        if k.endswith('.py'):ast.parse(v)


def test_scope_adapter_retains_original_train_partition():
    payload=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    namespace={};exec('from research.focus_extra_feature_scope import scope',namespace)
    assert namespace['scope'](payload)==scope(payload)
    assert not set(scope(payload)['fitting_stems'])&set(scope(payload)['unchanged_diagnostic_stems'])


def test_embedded_scope_imports_in_isolated_process(tmp_path):
    policy=scope((ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())
    nb,_=runpy.run_path(str(ROOT/'scripts/build-focus-extra-fit-features.py'))['build']({'contract':policy,'movies':[]})
    runtime=assignment(nb)
    for name in ('focus_adaptation_cache_contract.py','research/__init__.py','research/focus_extra_feature_scope.py','research/focus_extra_fit_scope.py','research/focus_adaptation_cache_contract.py','split.json'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(runtime[name],encoding='utf-8',newline='\n')
    code="import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from focus_adaptation_cache_contract import scope; print(json.dumps(scope((Path(sys.argv[1])/'split.json').read_bytes())))"
    completed=subprocess.run([sys.executable,'-I','-c',code,str(tmp_path)],cwd=tmp_path,timeout=20,capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert json.loads(completed.stdout)==policy
