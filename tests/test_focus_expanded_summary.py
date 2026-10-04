import ast
import json
from pathlib import Path
import runpy
import pytest

ROOT=Path(__file__).resolve().parents[1]
BUILD=runpy.run_path(str(ROOT/'scripts/build-focus-expanded-summary.py'))


def function(source,name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)


def test_original_forward_unchanged_and_no_training_or_diagnostic_call():
    source=BUILD['worker']();old=(ROOT/'scripts/train-focus-adaptation-head.py').read_text()
    assert ast.dump(function(source,'forward'))==ast.dump(function(old,'forward'))
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Name) and n.func.id=='evaluate' for n in calls)
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('AdamW','backward','step') for n in calls)
    assert "record['role']!='fitting'" in source
    assert "if not is_replay and replayed!=list(spec['replay_summaries'])" in source
    assert 'not np.array_equal(saved[k],v)' in source
    assert 'Empty-source known null requires separate declared handling' in source


def test_builder_pins_four_sources_and_bounded_offline_launcher():
    nb,meta=BUILD['build']({'contract':{'fitting_stems':[],'diagnostic_stems':[]}})
    assert len(meta['kernel_sources'])==4
    assert meta['kernel_sources'][-1]=='indarkarhana/biohub-focus-presence-summary-v1/1'
    assert meta['enable_gpu'] and meta['is_private'] and not meta['enable_internet'] and not meta['enable_tpu']
    final=''.join(nb['cells'][-1]['source'])
    assert "required_kernel('biohub-focus-extra-fit-features-v1','focus_extra_fit_features')" in final
    assert "required_kernel('biohub-focus-presence-summary-v1','focus_presence_summary')" in final
    assert '--extra-features-root' in final and '--replay-root' in final
    assert '3480-' in final
    for cell in nb['cells']:ast.parse(''.join(cell['source']))


def test_frozen_source_replacement_fails_closed():
    for source in ('absent','same same'):
        with pytest.raises(ValueError,match='exactly one'):
            BUILD['replace_once'](source,'same','new')
