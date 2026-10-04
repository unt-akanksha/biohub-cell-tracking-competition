"""No Torch required: inference provenance and immutable notebook wiring."""
import ast
import copy
import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/run-independent-selection-inference.py'))
PAIR = runpy.run_path(str(ROOT/'tests/test_owned_detector_fit_pair_receipts.py'))


def state_fixture(arm):
    _, _, results, split, probe = PAIR['fixture']()
    result = results[arm]
    state = dict(identity=result['identity'], step=1000,
                 input_hashes=result['input_hashes'], target_hashes=result['target_hashes'],
                 frozen_flow_model={'test_tensor': True},
                 optimizer={'state': {0: {'step': 1000}, 1: {'step': 1000}}})
    return state, split, probe


@pytest.mark.parametrize('arm', ['sparse', 'pu'])
def test_complete_owned_arm_requires_exact_training_and_probe(arm):
    state, split, probe = state_fixture(arm)
    M['check_completed_profile'](state, probe)
    receipt = M['owned_detector_fit_receipt'](state, probe)
    assert receipt['objective'] == arm and receipt['steps'] == 1000
    assert M['selection_scope'](state['identity'], split) == split['folds'][0]['selection']


@pytest.mark.parametrize('fault', ['partial', 'optimizer', 'prefix', 'targets', 'scope',
                                  'frozen', 'objective', 'probability', 'missing_probe', 'parent', 'flow', 'null'])
def test_owned_arm_rejects_incomplete_or_changed_training(fault):
    state, _, probe = state_fixture('pu')
    if fault == 'partial': state['step'] = 10
    if fault == 'optimizer': state['optimizer']['state'][0]['step'] = 999
    if fault == 'prefix': state['input_hashes'][0] = 'f'*64
    if fault == 'targets': state['target_hashes'].pop()
    if fault == 'scope': state['identity']['fine_tuning_stems'] = ['44b6_invalid']
    if fault == 'frozen': state['identity']['frozen_modules'] = ['flow']
    if fault == 'objective': state['identity']['owned_detector_objective'] = 'new'
    if fault == 'probability': state['identity']['teacher_peak_order'] = 'probabilities'
    if fault == 'missing_probe': probe = None
    if fault == 'parent': state['identity']['parent_identity'] = {}
    if fault == 'flow': state['identity']['frozen_flow_sha256'] = 'f'*64
    if fault == 'null': state['identity']['known_null'] = {}
    with pytest.raises(ValueError): M['check_completed_profile'](state, probe)


@pytest.mark.parametrize('arm', ['sparse', 'pu'])
def test_owned_builders_bind_verified_checkpoint_and_matching_scorer(monkeypatch, arm):
    report = PAIR['M']['verify'](*PAIR['fixture']())
    report_path = ROOT/'reports/experiments/owned-detector-fit-pair-v1-result.json'
    replacements = {report_path: json.dumps(report)}
    read_text, read_bytes = Path.read_text, Path.read_bytes
    monkeypatch.setattr(Path, 'read_text', lambda self,*a,**k:
                        replacements[self] if self in replacements else read_text(self,*a,**k))
    monkeypatch.setattr(Path, 'read_bytes', lambda self,*a,**k:
                        replacements[self].encode() if self in replacements else read_bytes(self,*a,**k))
    nb, meta, target = runpy.run_path(str(ROOT/'scripts/build-owned-detector-selection.py'))['build'](arm)
    launch = ''.join(nb['cells'][-1]['source'])
    assert f'owned_detector_fit_pair/outputs/{arm}/last.pt' in launch
    assert report['arms'][arm]['checkpoint_sha256'] in launch
    assert '--detector-spatial-tta' in launch and '--standalone-image-flow' in launch
    assert meta['kernel_sources'] == ['indarkarhana/biohub-owned-detector-fit-pair-v1/1']
    state, _, probe = state_fixture(arm)
    assert nb['metadata']['codex']['owned_detector_fit'] == M['owned_detector_fit_receipt'](state,probe)
    replacements[target/meta['code_file']] = json.dumps(nb)
    scored, metadata, _ = runpy.run_path(str(ROOT/'scripts/build-owned-detector-scoring.py'))['build'](arm)
    assert not metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['kernel_sources'] == [f'indarkarhana/biohub-owned-detector-{arm}-selection-v1/1']
    source = ''.join(scored['cells'][-1]['source'])
    assert f"p/'owned_detector_{arm}_selection'" in source
    for cell in scored['cells']: ast.parse(''.join(cell['source']))


def test_unregistered_trained_detector_receipt_rejected():
    module = runpy.run_path(str(ROOT/'tests/test_independent_selection_scoring.py'))
    manifest, terminal, codex, split = module['fixture']()
    state, _, probe = state_fixture('pu')
    manifest['owned_detector_fit'] = M['owned_detector_fit_receipt'](state, probe)
    with pytest.raises(ValueError, match='Unregistered'):
        module['M']['verify_manifest'](manifest, terminal, codex, split)
