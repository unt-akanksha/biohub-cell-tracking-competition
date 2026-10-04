import ast
import copy
import json
from pathlib import Path
import runpy
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_probe_scope_retains_full_parent_provenance_and_excludes_target():
    module=runpy.run_path(str(ROOT/'scripts/run-owned-detector-pu-probe.py'))
    split=json.loads((ROOT/'research/independent_real_baseline_v1_split.json').read_text())
    identity=dict(training_stems=split['folds'][0]['train'])
    assert module['scope'](identity,split)==identity['training_stems'][:4]
    bad=copy.deepcopy(identity); bad['training_stems'].append(split['folds'][0]['audit_order'][0])
    with pytest.raises(ValueError): module['scope'](bad,split)


def test_builder_keeps_original_weights_offline_and_ten_step_scope():
    nb,meta=runpy.run_path(str(ROOT/'scripts/build-owned-detector-pu-probe.py'))['build']()
    assert meta['kernel_sources']==['indarkarhana/biohub-image-motion-linker-v1/2']
    assert not meta['enable_internet'] and nb['metadata']['codex']['max_steps']==10
    assert nb['metadata']['codex']['authorized_for_submission'] is False
    source=''.join(nb['cells'][1]['source'])
    assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='runtime_sources')
    runtime=ast.literal_eval(assignment.value)
    assert 'owned_detector_pu.py' in runtime and 'spotiflow_biohub/pu_targets.py' in runtime
    assert "training_profile(profile,objective_name,logit_targets" in runtime['run_pilot.py']
    assert "default='probe'" in runtime['run_pilot.py']
    assert 'tests/test_owned_detector_pu_gradients.py' in runtime
    for cell in nb['cells']: ast.parse(''.join(cell['source']))
