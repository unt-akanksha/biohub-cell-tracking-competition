import ast
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_full_training_scope_is_exact_and_excludes_both_evaluation_sets():
    scope = runpy.run_path(str(ROOT/'scripts/run-independent-association-pilot.py'))['training_scope']
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    assert scope(split,'full') == split['folds'][0]['train']
    assert len(scope(split,'full')) == 120
    assert len(scope(split,'pilot')) == 4
    with pytest.raises(ValueError):
        scope(split,'arbitrary')
    split['folds'][0]['train'][5] = split['folds'][0]['selection'][0]
    with pytest.raises(ValueError,match='overlap'):
        scope(split,'full')


def test_broad_pair_is_sequential_bounded_and_keeps_small_gate():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-independent-motion-broad-pair.py'))['build']()
    launch = ''.join(nb['cells'][-1]['source'])
    assert "'--steps','1000','--training-scope','full'" in launch
    assert "runtime/'run_row_pair.py'" in launch
    assert meta['id'] == 'indarkarhana/biohub-independent-motion-broad-pair-v1'
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-real-pilot-v1/4']
    assert not meta['enable_internet']
    assert nb['metadata']['codex']['declared_budget_seconds'] == 3600
    source = ''.join(nb['cells'][1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    assert 'if step > 100 and (step100_smoke is None' in runtime['run_association.py']
    assert 'metadata_bytes > 4*1024**3' in runtime['run_association.py']
    assert "for arm in ('control','row'):" in runtime['run_row_pair.py']
    assert "!= 2*args.steps" in runtime['run_row_pair.py']
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))


@pytest.mark.parametrize('arm',['control','row'])
def test_selection_builder_binds_exact_arm_checkpoint(arm):
    build = runpy.run_path(str(ROOT/'scripts/build-independent-motion-broad-selection.py'))['build']
    nb,meta = build('a'*64,1,arm)
    launch = ''.join(nb['cells'][-1]['source'])
    assert f'independent_motion_broad_pair/outputs/{arm}/last.pt' in launch
    assert 'a'*64 in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-motion-broad-pair-v1/1']
    assert nb['metadata']['codex']['paired_training_arm'] == arm
    assert not meta['enable_internet']
    assert len(meta['title']) <= 50 and len(meta['id'].split('/')[1]) <= 50
