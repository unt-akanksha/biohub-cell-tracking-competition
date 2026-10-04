from copy import deepcopy
from pathlib import Path
import hashlib
import json
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT/'scripts/verify-independent-real-pilot.py'))


def row(step=10):
    return dict(step=step, edge_loss=.1, detection_loss=.2, elapsed_seconds=step,
                matched_training_nodes=1, annotated_training_nodes=10, training_node_recall=.1,
                supervised_training_pairs=1, optimizer_steps_max=step, grad_scaler_scale=1024.,
                max_detected_nodes=100, gpu_peak_allocated=[1000, 1000])


def test_history_requires_finite_losses_two_gpus_and_all_blocks():
    M['check_history']([row(), row(20)], 20)
    for key, value in [('edge_loss', float('nan')), ('max_detected_nodes', 2049),
                       ('gpu_peak_allocated', [1000]), ('elapsed_seconds', 0)]:
        bad = deepcopy(row()); bad[key] = value
        with pytest.raises(ValueError):
            M['check_history']([bad], 10)
    with pytest.raises(ValueError):
        M['check_history']([row(20)], 20)


def test_sources_are_bound_to_actual_launch_notebook():
    bundles = M['embedded_sources'](ROOT/'kaggle/biohub-independent-real-pilot-v1/biohub-independent-real-pilot-v1.ipynb')
    assert 'run_pilot.py' in bundles['runtime_sources']
    assert 'scripts/train_unet_transformer.py' in bundles['sources']


def test_missing_artifacts_fail_without_loading_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        M['verify'](tmp_path, ROOT/'kaggle/biohub-independent-real-pilot-v1/biohub-independent-real-pilot-v1.ipynb')


def make_complete_fixture(root):
    torch = pytest.importorskip('torch')
    notebook = ROOT/'kaggle/biohub-independent-real-pilot-v1/biohub-independent-real-pilot-v1.ipynb'
    bundles = M['embedded_sources'](notebook)
    for key, folder, filename in [('sources','repo','source_hashes.json'),
            ('runtime_sources','runtime','runtime_hashes.json')]:
        hashes = {}
        for name, content in bundles[key].items():
            target = root/folder/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode())
            hashes[name] = hashlib.sha256(content.encode()).hexdigest()
        (root/filename).write_text(json.dumps(hashes))
    split = json.loads(bundles['runtime_sources']['split.json'])
    identity = dict(run_id='independent-real-pilot-v1',
        max_steps=json.loads(notebook.read_text())['metadata']['codex'].get('max_steps',100),
        training_stems=split['folds'][0]['train'][:4],
        held_out_embryo=split['folds'][0]['held_out_embryo'], seed=20260909,
        empty_attention_guard=True,
        detector_warmup_steps=50, detector_initial_probability=.01,
        manifest_sha256=hashlib.sha256(bundles['runtime_sources']['split.json'].encode()).hexdigest(),
        target_audit_opened=False, public_checkpoint_loaded=False, authorized_for_submission=False)
    output = root/'outputs'; output.mkdir()
    warmup = [dict(step=i, loss=.1) for i in (10,20,30,40,50)]
    (output/'warmup_history.json').write_text(json.dumps(warmup))
    state = dict(identity=identity, step=10, warmup_steps=50, model={'weight':torch.tensor([1.])},
        optimizer={'state':{0:{'step':torch.tensor(10)}}}, scaler={}, torch_rng=torch.get_rng_state(),
        cuda_rng=[torch.get_rng_state(),torch.get_rng_state()], loader_rng=torch.get_rng_state(), python_rng=[], numpy_rng={})
    torch.save(state, output/'last.pt')
    result = dict(identity, status='completed_pilot', trained_steps=10, history=[row()],
        warmup_history=warmup,
        checkpoint_sha256=M['sha'](output/'last.pt'), selection_opened=False,
        production_promotion_authorized=False)
    for name, contents in [('identity.json', identity), ('result.json',result), ('history.json',[row()])]:
        (output/name).write_text(json.dumps(contents))
    (root/'launcher_terminal.json').write_text(json.dumps(dict(status='completed',
        run_id=identity['run_id'], submission_performed=False, declared_budget_seconds=3600,
        elapsed_seconds=60)))
    return notebook


def test_complete_artifacts_are_verified_without_authorizing_promotion(tmp_path):
    notebook = make_complete_fixture(tmp_path)
    result = M['verify'](tmp_path, notebook)
    assert result['trained_steps'] == 10 and result['optimizer_steps_min'] == 10
    assert result['authorized_for_submission'] is False
    assert result['longer_gpu_run_authorized'] is False


@pytest.mark.parametrize('mutation', ['checkpoint', 'runtime', 'training_scope'])
def test_complete_fixture_rejects_tampering(tmp_path, mutation):
    notebook = make_complete_fixture(tmp_path)
    if mutation == 'checkpoint':
        (tmp_path/'outputs/last.pt').write_bytes(b'corrupt')
    elif mutation == 'runtime':
        (tmp_path/'runtime/run_pilot.py').write_text('changed')
    else:
        path = tmp_path/'outputs/result.json'
        result = json.loads(path.read_text())
        result['target_audit_opened'] = True
        path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        M['verify'](tmp_path, notebook)


@pytest.mark.parametrize('mutation', [None, 'absent', 'cpu', 'corrupt'])
def test_extended_profile_requires_real_gpu_smoke_checkpoint(tmp_path, mutation):
    import torch
    notebook = make_complete_fixture(tmp_path)
    output = tmp_path/'outputs'
    result = json.loads((output/'result.json').read_text())
    if result.get('max_steps', 100) < 110:
        pytest.skip('Extended launch notebook required')
    state = torch.load(output/'last.pt', weights_only=True)
    state['step'] = 100
    torch.save(state, output/'smoke_checkpoint.pt')
    smoke = dict(status='passed', device='cuda', checkpoint_step=100, frames=3,
        movie=result['training_stems'][0], strict_reload=True, geff_round_trip=True,
        checkpoint_sha256=M['sha'](output/'smoke_checkpoint.pt'),
        scorer_counts=dict(edge_tp=0,edge_fp=0,edge_fn=1,division_tp=0,division_fp=0,division_fn=0,num_pred_nodes=0))
    state['step'] = 110
    state['optimizer']['state'][0]['step'] = torch.tensor(110)
    torch.save(state, output/'last.pt')
    result.update(trained_steps=110, history=[row(i) for i in range(10,111,10)],
                  checkpoint_sha256=M['sha'](output/'last.pt'), gpu_smoke=smoke)
    if mutation == 'absent':
        result.pop('gpu_smoke')
    elif mutation == 'cpu':
        smoke['device'] = 'cpu'
    elif mutation == 'corrupt':
        (output/'smoke_checkpoint.pt').write_bytes(b'bad')
    (output/'result.json').write_text(json.dumps(result))
    (output/'history.json').write_text(json.dumps(result['history']))
    (output/'gpu_smoke.json').write_text(json.dumps(smoke))
    if mutation:
        with pytest.raises(ValueError):
            M['verify'](tmp_path, notebook)
    else:
        assert M['verify'](tmp_path, notebook)['trained_steps'] == 110
