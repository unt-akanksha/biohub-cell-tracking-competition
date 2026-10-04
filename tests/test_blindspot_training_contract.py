from pathlib import Path
import ast
import json
import runpy
import pytest
from research.blindspot_training_contract import identity,origin,proxy_gate,PATCH,FRAMES

ROOT=Path(__file__).resolve().parents[1]


def policy():
    return identity((ROOT/'research/independent_real_baseline_v1_split.json').read_bytes())


def test_scope_normalization_and_reproducible_native_sampling():
    p=policy()
    assert len(p['fitting_stems'])==96 and len(p['diagnostic_stems'])==24
    assert not set(p['fitting_stems'])&set(p['diagnostic_stems'])
    for s in p['fitting_stems']+p['diagnostic_stems']:
        for t in FRAMES:
            start=origin(s,t)
            assert start==origin(s,t)
            assert all(0<=a<=n-k for a,n,k in zip(start,(64,256,256),PATCH))
    assert not p['ground_truth_opened'] and not p['selection_opened']


def test_changed_split_or_frame_rejected():
    with pytest.raises(ValueError): identity(b'{}')
    with pytest.raises(ValueError): origin('6bba_test',50)


def test_proxy_requires_pooled_gain_and_eighteen_movies():
    stems=policy()['diagnostic_stems']
    rows=[dict(stem=s,pixels=135000,model_sse=1.,neighbor_sse=2.) for s in stems]
    assert proxy_gate(rows,stems)['proxy_gate_passed']
    for r in rows[:7]: r['model_sse']=2.
    assert not proxy_gate(rows,stems)['proxy_gate_passed']
    rows[0]['model_sse']=float('nan')
    with pytest.raises(ValueError): proxy_gate(rows,stems)


def test_builder_exact_model_and_bounded_offline_training():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-blindspot-real-probe.py'))['build']()
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu']
    assert not meta['kernel_sources'] and nb['metadata']['codex']['declared_budget_seconds']==3600
    source=''.join(nb['cells'][1]['source'])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    bundle=ast.literal_eval(node.value)
    assert bundle['blindspot_restoration.py']==(ROOT/'research/blindspot_restoration.py').read_text()
    assert 'require_tracks=True' not in bundle['run_pilot.py']
    assert '--steps' in ''.join(nb['cells'][-1]['source'])
