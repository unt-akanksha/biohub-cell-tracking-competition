import ast
import json
from pathlib import Path
import runpy
import pytest
from research.focus_extra_fit_scope import scope

ROOT=Path(__file__).resolve().parents[1]


def test_fixed_order_and_unchanged_diagnostic():
    payload=(ROOT/'research/independent_real_baseline_v1_split.json').read_bytes();policy=scope(payload)
    fold=json.loads(payload)['folds'][0]
    assert len(policy['fitting_stems'])==8
    assert not set(policy['fitting_stems'])&set(fold['train'][::5]+fold['selection']+fold['audit_order']+policy['previous_fitting_stems']+policy['replay_stems'])
    assert policy['unchanged_diagnostic_stems']==['6bba_b204cac7','6bba_2646afc7','6bba_8b7818bf','6bba_312f0dc3']
    with pytest.raises(ValueError):scope(payload+b' ')


def test_exact_runtime_scope_only_replacement():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-focus-extra-fit-cache.py'))['build']()
    assert meta['enable_gpu'] and not meta['enable_internet'] and not meta['enable_tpu'] and meta['is_private']
    assert meta['kernel_sources']==['indarkarhana/biohub-focus-source-probe-v1/1']
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    source='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'focus_extra_fit_cache_terminal.json' in source
    assert 'Full source run failed exact small-probe replay' in source
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
