import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fork16_portable_builder',ROOT/'scripts/build-trajectory-event-fork16-vectorized-portable-v1.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def runtime():
    source=builder.portable_source();namespace={'__name__':'portable_test'}
    exec(compile(source,'portable_test','exec'),namespace)
    return namespace


def test_no_project_imports_or_training_entrypoints():
    tree=ast.parse(builder.portable_source())
    assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('research') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('fit','prepare','train') for n in ast.walk(tree))


def test_offline_raw_input_interface_uses_expanded_vocabulary_and_identity_guard():
    model=runtime()
    graph=dict(nodes={str(i):dict(node_id=i,t=int(i>0),z=0,y=0,x=i) for i in range(11)},
               edges=[dict(source_id=0,target_id=1)])
    coords=np.array([[n[k] for k in ('t','z','y','x')] for n in graph['nodes'].values()])
    raw_edges=np.empty((0,4));weights=np.zeros(30)
    result,details=model['refine'](graph,graph,coords,raw_edges,weights)
    assert result==graph and details['processed_frames']==1
    assert details['frames'][0]['options']==66
    weights[19]=100
    result,details=model['refine'](graph,graph,coords,raw_edges,weights)
    assert result['nodes']==graph['nodes'] and len(result['edges'])==2
    changed=coords.copy();changed[0,3]+=1
    with pytest.raises(ValueError,match='Raw detector identity changed'):
        model['refine'](graph,graph,changed,raw_edges,weights)
