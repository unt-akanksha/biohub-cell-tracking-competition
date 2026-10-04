import ast
import copy
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = runpy.run_path(str(ROOT/'scripts/score-causal-motion-selection.py'))


def test_frozen_fit_rejects_contaminated_membership_and_scope():
    split = (ROOT/'research/independent_real_baseline_v1_split.json').read_bytes()
    receipt = json.loads((ROOT/'reports/experiments/independent-motion-persistence-training.json').read_text())
    assert WORKER['validate_fit'](receipt, split) == receipt['model']
    for key, value in [('selection_opened', True), ('fitting_stems', receipt['fitting_stems'][1:]),
                       ('split_sha256', '0'*64), ('diagnostic_stems', receipt['fitting_stems'][:24])]:
        bad = copy.deepcopy(receipt)
        bad[key] = value
        with pytest.raises(ValueError):
            WORKER['validate_fit'](bad, split)


def test_cpu_builder_embeds_exact_native_input_and_frozen_fit():
    nb, meta = runpy.run_path(str(ROOT/'scripts/build-causal-motion-selection.py'))['build']()
    assert meta['enable_gpu'] is False and meta['enable_internet'] is False
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-selection-v1/1']
    source = ''.join(nb['cells'][-1]['source'])
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'scoring_sources')
    bundle = ast.literal_eval(assignment.value)
    assert bundle['selection_launch.ipynb'] == (ROOT/'kaggle/biohub-independent-joint-selection-v1/biohub-independent-joint-selection-v1.ipynb').read_text()
    path = 'reports/experiments/independent-motion-persistence-training.json'
    assert bundle[path] == (ROOT/path).read_text(encoding='utf-8')
    tail = source[source.index('\nimport runpy'):]
    assert tail.index('if gate.returncode != 0') < tail.index('result = runpy.run_path')
    assert "scoring/'scripts/score-causal-motion-selection.py'" in tail


def test_static_control_is_cpu_and_uses_same_frozen_nodes():
    nb, meta = runpy.run_path(str(ROOT/'scripts/build-joint-static-motion-selection.py'))['build']()
    assert meta['enable_gpu'] is False and meta['enable_internet'] is False
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-joint-selection-v1/1']
    source = ''.join(nb['cells'][-1]['source'])
    tail = source[source.index('\nimport runpy'):]
    assert "scoring/'scripts/score-independent-motion-prior.py'" in tail
    assert 'tests/test_independent_motion_prior.py' in tail
    assert "result['run_id'] = 'joint-static-motion-selection-v1'" in tail
    assert "work/'static_motion_score.json'" in tail
