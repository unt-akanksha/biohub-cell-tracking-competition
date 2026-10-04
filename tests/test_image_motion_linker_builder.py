import argparse
import ast
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT/'scripts/build-image-motion-linker.py'))['build']


@pytest.mark.parametrize('steps',[100,1000])
def test_linker_builder_preserves_both_frozen_models_and_gates(steps):
    nb,meta = BUILD(steps)
    assert meta['enable_gpu'] and not meta['enable_internet']
    assert meta['kernel_sources'] == ['indarkarhana/biohub-independent-known-null-v1/2',
                                      'indarkarhana/biohub-backward-flow-fit-v1/1']
    assert nb['metadata']['codex']['max_steps'] == steps
    launch = ''.join(nb['cells'][-1]['source'])
    assert "'--image-motion','--flow-checkpoint'" in launch
    assert "'--known-null'" not in launch and "'--joint'" not in launch
    assert 'independent_known_null/outputs/last.pt' in launch
    assert 'backward_flow_fit/outputs/last.pt' in launch
    assert 'test_image_motion_residual.py' in launch and 'test_sparse_parent_missing_null.py' in launch
    assignment = next(n for n in ast.parse(''.join(nb['cells'][1]['source'])).body
        if isinstance(n,ast.Assign) and n.targets[0].id == 'runtime_sources')
    runtime = ast.literal_eval(assignment.value)
    assert 'embedded_flow(state,device)' in runtime['real_checkpoint_gpu_smoke.py']
    assert "payload['frozen_flow_model'] = flow_core.state_dict()" in runtime['run_association.py']
    assert "flow_hash(flow_core) == identity['frozen_flow_sha256']" in runtime['run_association.py']
    assert 'run_id=\'image-motion-linker-v1\'' in runtime['run_association.py']
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))


def test_image_profile_rejects_incompatible_or_wrong_initialization():
    main = runpy.run_path(str(ROOT/'scripts/run-independent-association-pilot.py'))['main']
    base = dict(image_motion=True,known_null=False,division_specialist=False,joint=False,row_negatives=False,
                motion_residual=True,training_scope='full',steps=100,
                sha256='b64254aece75ef709e517621757a5313ac665e0803050ce4f13fa2a76602d6ef')
    for changes in (dict(known_null=True),dict(division_specialist=True),dict(joint=True),
                    dict(sha256='a'*64),dict(training_scope='pilot')):
        with pytest.raises(ValueError):
            main(argparse.Namespace(**(base|changes)))
