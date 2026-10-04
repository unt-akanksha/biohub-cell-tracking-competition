import ast
import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = runpy.run_path(str(ROOT/'scripts/run-association-calibration.py'))


def test_scope_is_training_only_and_probe_is_nested():
    split = json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    fit,diag = RUN['scope'](split,'full')
    probe_fit,probe_diag = RUN['scope'](split,'probe')
    assert len(fit)==96 and len(diag)==24 and not set(fit)&set(diag)
    assert probe_fit==fit[:4] and probe_diag==diag[:2]
    bad = copy.deepcopy(split)
    bad['folds'][0]['selection'].append(fit[0])
    with pytest.raises(ValueError):
        RUN['scope'](bad,'full')


def test_builder_has_exact_checkpoint_offline_budget_and_no_training():
    nb,meta = runpy.run_path(str(ROOT/'scripts/build-association-calibration-probe.py'))['build']()
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2']
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert nb['metadata']['codex']['declared_budget_seconds']==3600
    launch = ''.join(nb['cells'][-1]['source'])
    assert "'--checkpoint', str(checkpoint)" in launch and '--steps' not in launch
    assert 'image_motion_linker/outputs/last.pt' in launch
    assignment = next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    source = runtime['run_pilot.py']
    assert '.requires_grad_(False).eval()' in source and 'optimizer.step' not in source
    assert RUN['CHECKPOINT_SHA'] in source and 'allow_pickle=False' in source
    assert 'association_calibration.py' in runtime and 'annotated_missing_parent.py' in runtime
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
